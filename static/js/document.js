document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file');
    const filesInput = document.getElementById('files');

    if (fileInput) {
        fileInput.addEventListener('dragover', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = 'copy';
        });

        fileInput.addEventListener('drop', (event) => {
            event.preventDefault();
            event.stopPropagation();
            fileInput.files = event.dataTransfer.files;
        });
    }

    if (filesInput) {
        filesInput.addEventListener('dragover', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = 'copy';
        });

        filesInput.addEventListener('drop', (event) => {
            event.preventDefault();
            event.stopPropagation();
            filesInput.files = event.dataTransfer.files;
        });
    }
});
