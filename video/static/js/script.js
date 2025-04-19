document.addEventListener('DOMContentLoaded', function () {
    const videoGrid = document.getElementById('video-grid');
    const pollInterval = 2000;

    function createVideoElement(clientId) {
        const container = document.createElement('div');
        container.className = 'video-container';
        container.id = `container-${clientId}`;

        const title = document.createElement('h3');
        title.innerText = `Client: ${clientId}`;
        container.appendChild(title);

        const img = document.createElement('img');
        img.id = `video-${clientId}`;
        img.src = `/video_feed/${encodeURIComponent(clientId)}?t=${Date.now()}`;
        img.alt = `Stream ${clientId}`;
        img.className = 'video-stream';
        container.appendChild(img);

        const controls = document.createElement('div');
        controls.className = 'controls';

        const select = document.createElement('select');
        ['Select Gender', 'Male', 'Female'].forEach(opt => {
            const option = document.createElement('option');
            option.value = opt;
            option.innerText = opt;
            select.appendChild(option);
        });

        const toggleBtn = document.createElement('button');
        toggleBtn.innerText = 'Activate Detection';
        toggleBtn.onclick = function () {
            const gender = select.value;
            if (gender !== 'Male' && gender !== 'Female') {
                alert('Please select a valid gender.');
                return;
            }

            fetch(`/toggle_detection/${clientId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gender })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status) {
                    alert(data.status);
                } else if (data.error) {
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(() => alert('Error toggling detection'));
        };

        const deactivateBtn = document.createElement('button');
        deactivateBtn.innerText = 'Deactivate Detection';
        deactivateBtn.onclick = function () {
            fetch(`/toggle_detection/${clientId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gender: 'None' })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status) {
                    alert(data.status);
                } else if (data.error) {
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(() => alert('Error deactivating detection'));
        };

        const removeBtn = document.createElement('button');
        removeBtn.innerText = 'Remove';
        removeBtn.onclick = () => container.remove();

        controls.appendChild(select);
        controls.appendChild(toggleBtn);
        controls.appendChild(deactivateBtn);
        controls.appendChild(removeBtn);
        container.appendChild(controls);

        return container;
    }

    function checkForStreams() {
        fetch('/active_streams')
            .then(response => response.json())
            .then(data => {
                const activeClients = data.clients || [];

                document.querySelectorAll('.video-container').forEach(container => {
                    const clientId = container.id.replace('container-', '');
                    if (!activeClients.includes(clientId)) {
                        container.remove();
                    }
                });

                activeClients.forEach(clientId => {
                    if (!document.getElementById(`container-${clientId}`)) {
                        videoGrid.appendChild(createVideoElement(clientId));
                    }

                    const img = document.getElementById(`video-${clientId}`);
                    if (img) {
                        img.src = `/video_feed/${encodeURIComponent(clientId)}?t=${Date.now()}`;
                    }
                });
            })
            .catch(console.error);
    }

    checkForStreams();
    setInterval(checkForStreams, pollInterval);
});
