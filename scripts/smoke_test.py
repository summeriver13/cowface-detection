"""端到端冒烟测试：用 Flask test client 跑一遍页面、接口、CSRF 与权限。

单独用一个临时 SQLite，不碰 instance/ 下的真实库；不需要模型权重（识别器是惰性加载的）。
用法：python scripts/smoke_test.py
"""

import io
import os
import pathlib
import re
import sys
import tempfile

TOKEN_PATTERN = re.compile(r'name="csrf-token" content="([^"]+)"')
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    if condition:
        print(f"  ok    {name}")
        return
    FAILURES.append(name)
    print(f"  FAIL  {name}  -> {detail}")


def fetch_token(client) -> str:
    page = client.get("/login").get_data(as_text=True)
    match = TOKEN_PATTERN.search(page)
    if match is None:
        raise RuntimeError("登录页没有 csrf meta，模板没渲染出来")
    return match.group(1)


def main() -> int:
    os.environ["DATABASE_URL"] = f"sqlite:///{pathlib.Path(tempfile.mkdtemp()) / 'smoke.db'}"
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    from app import create_app

    app = create_app()
    client = app.test_client()

    check("未登录访问首页跳转", client.get("/").status_code == 302)
    check("未登录访问 dashboard 跳转", client.get("/dashboard").status_code == 302)

    token = fetch_token(client)
    check("无 CSRF token 的 POST 被拒", client.post("/login", data={"username": "a", "password": "b"}).status_code == 400)

    check("注册", client.post("/register", data={"username": "smoke", "password": "pw", "csrf_token": token}).status_code == 302)
    check("登录", client.post("/login", data={"username": "smoke", "password": "pw", "csrf_token": token}).status_code == 302)

    for path in ("/dashboard", "/cattle", "/detect", "/train", "/train/status"):
        check(f"GET {path}", client.get(path).status_code == 200)

    check("接口无 token 被拒", client.post("/api/cattle", data={"name": "x"}).status_code == 400)
    created = client.post("/api/cattle", data={"name": "Cow (1)"}, headers={"X-CSRF-Token": token})
    check("新建牛只", created.status_code == 201, created.get_json())
    cattle_id = created.get_json()["data"]["cattle_id"]

    renamed = client.patch(f"/api/cattle/{cattle_id}", json={"name": "Cow (2)"}, headers={"X-CSRF-Token": token})
    check("改名", renamed.status_code == 200 and renamed.get_json()["data"]["name"] == "Cow (2)", renamed.get_json())
    check("空名被拒", client.post("/api/cattle", data={"name": ""}, headers={"X-CSRF-Token": token}).status_code == 400)
    check(
        "改名不存在的牛只 404",
        client.patch("/api/cattle/999999", json={"name": "z"}, headers={"X-CSRF-Token": token}).status_code == 404,
    )

    trainings = client.get("/api/train")
    check("训练列表", trainings.status_code == 200 and trainings.get_json()["data"]["trainings"] == [])
    invalid = client.post(
        "/api/train",
        data={"model_type": "zz", "epochs": "5", "batch_size": "2", "learning_rate": "0.001"},
        headers={"X-CSRF-Token": token},
    )
    check("非法模型类型被拒", invalid.status_code == 400, invalid.get_json())

    check("检测：没有文件 400", client.post("/api/detect", headers={"X-CSRF-Token": token}).status_code == 400)
    unsupported = client.post(
        "/api/detect",
        data={"image": (io.BytesIO(b"not an image"), "x.txt")},
        headers={"X-CSRF-Token": token},
        content_type="multipart/form-data",
    )
    check("检测：非图片 400", unsupported.status_code == 400, unsupported.get_json())

    missing = client.get("/api/does-not-exist")
    check("未知接口返回 JSON 信封", missing.status_code == 404 and missing.get_json()["ok"] is False, missing.get_json())

    check("删除牛只", client.delete(f"/api/cattle/{cattle_id}", headers={"X-CSRF-Token": token}).status_code == 200)
    check("匿名访问接口 401", app.test_client().get("/api/train").status_code == 401)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} 项失败：" + "，".join(FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
