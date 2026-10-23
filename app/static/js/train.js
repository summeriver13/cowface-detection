(function () {
    const form = document.getElementById("train-form");
    if (!form) {
        return;
    }

    const errorBox = document.getElementById("train-error");
    const submit = form.querySelector('button[type="submit"]');

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        errorBox.classList.add("d-none");
        submit.disabled = true;
        try {
            await api.request(form.dataset.endpoint, { method: "POST", formData: new FormData(form) });
            window.location.href = form.dataset.next;
        } catch (error) {
            errorBox.textContent = error.message;
            errorBox.classList.remove("d-none");
            submit.disabled = false;
        }
    });
})();
