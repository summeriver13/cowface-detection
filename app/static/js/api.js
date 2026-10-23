(function () {
    const meta = document.querySelector('meta[name="csrf-token"]');
    const token = meta ? meta.content : "";

    function notify(message, variant) {
        const area = document.getElementById("toast-area");
        if (!area) {
            window.alert(message);
            return;
        }
        const alert = document.createElement("div");
        alert.className = "alert alert-" + (variant || "info") + " alert-dismissible fade show mt-3";
        alert.setAttribute("role", "alert");
        alert.append(document.createTextNode(message));
        const close = document.createElement("button");
        close.type = "button";
        close.className = "btn-close";
        close.setAttribute("data-bs-dismiss", "alert");
        close.setAttribute("aria-label", "关闭");
        alert.append(close);
        area.append(alert);
        window.setTimeout(function () {
            alert.remove();
        }, 6000);
    }

    async function request(url, options) {
        const settings = options || {};
        const init = {
            method: settings.method || "GET",
            headers: { Accept: "application/json", "X-CSRF-Token": token }
        };
        if (settings.formData) {
            init.body = settings.formData;
        } else if (settings.json !== undefined) {
            init.headers["Content-Type"] = "application/json";
            init.body = JSON.stringify(settings.json);
        }

        const response = await fetch(url, init);
        let payload = null;
        try {
            payload = await response.json();
        } catch (error) {
            payload = null;
        }
        if (!response.ok || !payload || payload.ok !== true) {
            const message = payload && payload.error && payload.error.message
                ? payload.error.message
                : "请求失败（HTTP " + response.status + "）";
            throw new Error(message);
        }
        return payload.data;
    }

    window.api = { request: request, notify: notify };
})();
