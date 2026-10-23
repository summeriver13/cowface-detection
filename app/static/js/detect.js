(function () {
    const page = document.getElementById("detect-page");
    if (!page) {
        return;
    }

    const input = document.getElementById("image-input");
    const button = document.getElementById("detect-button");
    const errorBox = document.getElementById("detect-error");
    const loading = document.getElementById("detect-loading");
    const result = document.getElementById("detect-result");
    const preview = document.getElementById("detect-preview");
    const list = document.getElementById("detect-list");
    const savedNote = document.getElementById("detect-saved");

    function showError(message) {
        errorBox.textContent = message;
        errorBox.classList.toggle("d-none", !message);
    }

    function renderDetections(detections) {
        list.replaceChildren();
        if (!detections.length) {
            const item = document.createElement("li");
            item.className = "list-group-item text-muted";
            item.textContent = "未检测到牛脸";
            list.append(item);
            return;
        }
        detections.forEach(function (detection) {
            const item = document.createElement("li");
            item.className = "list-group-item";
            const detectionConf = (detection.detection_conf * 100).toFixed(1);
            const classConf = (detection.class_conf * 100).toFixed(1);
            item.textContent = detection.class + " · 检测 " + detectionConf + "% · 识别 " + classConf + "%";
            list.append(item);
        });
    }

    button.addEventListener("click", async function () {
        showError("");
        const file = input.files[0];
        if (!file) {
            showError("请先选择图片");
            return;
        }

        const data = new FormData();
        data.append("image", file);
        button.disabled = true;
        loading.classList.remove("d-none");
        result.classList.add("d-none");

        try {
            const payload = await api.request(page.dataset.endpoint, { method: "POST", formData: data });
            preview.src = "data:image/jpeg;base64," + payload.image;
            renderDetections(payload.detections);
            savedNote.textContent = payload.saved_records
                ? "已写入 " + payload.saved_records + " 条识别记录"
                : "没有同名牛只，本次未入库";
            savedNote.classList.remove("d-none");
            result.classList.remove("d-none");
        } catch (error) {
            showError(error.message);
        } finally {
            button.disabled = false;
            loading.classList.add("d-none");
        }
    });
})();
