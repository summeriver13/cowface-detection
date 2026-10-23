(function () {
    const page = document.getElementById("training-status-page");
    if (!page) {
        return;
    }

    const POLL_MS = 3000;
    const body = document.getElementById("training-rows");
    const errorBox = document.getElementById("status-error");
    const labels = window.MODEL_TYPE_LABELS || {};

    function formatMetric(value) {
        return value === null || value === undefined ? "-" : Number(value).toFixed(3);
    }

    function formatTime(iso) {
        return iso ? iso.replace("T", " ").slice(0, 19) : "-";
    }

    function cell(text, className) {
        const node = document.createElement("td");
        node.textContent = text;
        if (className) {
            node.className = className;
        }
        return node;
    }

    function buildRow(training) {
        const row = document.createElement("tr");
        row.dataset.trainingId = training.training_id;
        row.append(cell(labels[training.model_type] || training.model_type));
        row.append(cell(formatTime(training.start_time), "text-nowrap"));

        const statusCell = document.createElement("td");
        const badge = document.createElement("span");
        badge.className = "badge status-badge status-" + training.status;
        badge.textContent = training.status;
        statusCell.append(badge);
        row.append(statusCell);

        row.append(cell(formatMetric(training.map50)));
        row.append(cell(formatMetric(training.map50_95)));
        row.append(cell(formatMetric(training.precision)));
        row.append(cell(formatMetric(training.recall)));

        const actionCell = document.createElement("td");
        actionCell.className = "text-end";
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "btn btn-sm btn-outline-danger js-delete";
        remove.textContent = "删除";
        actionCell.append(remove);
        row.append(actionCell);
        return row;
    }

    function emptyRow() {
        const row = document.createElement("tr");
        const node = cell("暂无训练记录", "text-center text-muted py-4");
        node.colSpan = 8;
        row.append(node);
        return row;
    }

    function render(trainings) {
        body.replaceChildren();
        if (!trainings.length) {
            body.append(emptyRow());
            return false;
        }
        trainings.forEach(function (training) {
            body.append(buildRow(training));
        });
        return trainings.some(function (training) {
            return training.status === "training";
        });
    }

    async function refresh() {
        try {
            const data = await api.request(page.dataset.listUrl);
            const stillRunning = render(data.trainings);
            errorBox.classList.add("d-none");
            if (stillRunning) {
                window.setTimeout(refresh, POLL_MS);
            }
        } catch (error) {
            errorBox.textContent = error.message;
            errorBox.classList.remove("d-none");
        }
    }

    body.addEventListener("click", async function (event) {
        const button = event.target.closest(".js-delete");
        if (!button) {
            return;
        }
        const trainingId = button.closest("tr").dataset.trainingId;
        try {
            await api.request(page.dataset.deleteBase + trainingId, { method: "DELETE" });
            await refresh();
        } catch (error) {
            errorBox.textContent = error.message;
            errorBox.classList.remove("d-none");
        }
    });

    refresh();
})();
