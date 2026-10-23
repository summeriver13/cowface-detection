(function () {
    const page = document.getElementById("cattle-page");
    if (!page) {
        return;
    }

    const errorBox = document.getElementById("cattle-error");
    const form = document.getElementById("add-cattle-form");
    const tableBody = page.querySelector("tbody");

    function showError(message) {
        errorBox.textContent = message;
        errorBox.classList.remove("d-none");
    }

    function clearError() {
        errorBox.textContent = "";
        errorBox.classList.add("d-none");
    }

    function toggleEdit(row, editing) {
        row.querySelector(".js-name").classList.toggle("d-none", editing);
        row.querySelector(".js-name-input").classList.toggle("d-none", !editing);
        row.querySelector(".js-rename").classList.toggle("d-none", editing);
        row.querySelector(".js-save").classList.toggle("d-none", !editing);
        if (editing) {
            row.querySelector(".js-name-input").focus();
        }
    }

    async function saveName(row, cattleId) {
        const name = row.querySelector(".js-name-input").value.trim();
        if (!name) {
            showError("牛只名称不能为空");
            return;
        }
        try {
            await api.request(page.dataset.renameBase + cattleId, { method: "PATCH", json: { name: name } });
            row.querySelector(".js-name").textContent = name;
            toggleEdit(row, false);
            api.notify("已保存", "success");
        } catch (error) {
            showError(error.message);
        }
    }

    async function removeCattle(cattleId) {
        if (!window.confirm("确定删除这只牛？相关识别记录会一起删除。")) {
            return;
        }
        try {
            await api.request(page.dataset.deleteBase + cattleId, { method: "DELETE" });
            window.location.reload();
        } catch (error) {
            showError(error.message);
        }
    }

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        clearError();
        const data = new FormData(form);
        if (!String(data.get("name") || "").trim()) {
            showError("牛只名称不能为空");
            return;
        }
        try {
            await api.request(page.dataset.createUrl, { method: "POST", formData: data });
            window.location.reload();
        } catch (error) {
            showError(error.message);
        }
    });

    tableBody.addEventListener("click", function (event) {
        const button = event.target.closest("button");
        if (!button) {
            return;
        }
        const row = button.closest("tr");
        const cattleId = row.dataset.cattleId;
        clearError();

        if (button.classList.contains("js-rename")) {
            toggleEdit(row, true);
        } else if (button.classList.contains("js-save")) {
            saveName(row, cattleId);
        } else if (button.classList.contains("js-delete")) {
            removeCattle(cattleId);
        } else if (button.classList.contains("js-open")) {
            window.location.href = page.dataset.detailBase + cattleId;
        }
    });
})();
