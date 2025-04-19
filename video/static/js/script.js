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

        // Status indicators
        const statusIndicators = document.createElement('div');
        statusIndicators.className = 'status-indicators';
        
        const motionStatus = document.createElement('div');
        motionStatus.id = `motion-status-${clientId}`;
        motionStatus.className = 'status-indicator';
        motionStatus.innerHTML = '<span class="indicator">⚪</span> Motion';
        statusIndicators.appendChild(motionStatus);
        
        const fightStatus = document.createElement('div');
        fightStatus.id = `fight-status-${clientId}`;
        fightStatus.className = 'status-indicator';
        fightStatus.innerHTML = '<span class="indicator">⚪</span> Fight';
        statusIndicators.appendChild(fightStatus);
        
        const genderStatus = document.createElement('div');
        genderStatus.id = `gender-status-${clientId}`;
        genderStatus.className = 'status-indicator';
        genderStatus.innerHTML = '<span class="indicator">⚪</span> Gender Detection';
        statusIndicators.appendChild(genderStatus);
        
        container.appendChild(statusIndicators);

        const controls = document.createElement('div');
        controls.className = 'controls';

        // Mode selection
        const modeSelect = document.createElement('select');
        modeSelect.id = `mode-${clientId}`;
        ['Select Mode', 'Motion Detection', 'Gender Detection'].forEach((opt, idx) => {
            const option = document.createElement('option');
            option.value = idx === 0 ? '' : idx === 1 ? 'motion' : 'gender';
            option.innerText = opt;
            modeSelect.appendChild(option);
        });
        controls.appendChild(modeSelect);

        // Gender selection (only shown when gender detection is selected)
        const genderSelect = document.createElement('select');
        genderSelect.id = `gender-${clientId}`;
        genderSelect.style.display = 'none';
        ['Select Gender', 'Male', 'Female'].forEach(opt => {
            const option = document.createElement('option');
            option.value = opt === 'Select Gender' ? '' : opt;
            option.innerText = opt;
            genderSelect.appendChild(option);
        });
        controls.appendChild(genderSelect);

        // Show/hide gender select based on mode selection
        modeSelect.onchange = function() {
            genderSelect.style.display = this.value === 'gender' ? 'inline-block' : 'none';
        };

        const toggleBtn = document.createElement('button');
        toggleBtn.innerText = 'Activate Detection';
        toggleBtn.className = 'activate-btn';
        toggleBtn.onclick = function () {
            const mode = modeSelect.value;
            if (!mode) {
                alert('Please select a detection mode.');
                return;
            }

            let requestBody = {
                mode: mode,
                active: true
            };

            if (mode === 'gender') {
                const gender = genderSelect.value;
                if (!gender) {
                    alert('Please select a gender to detect.');
                    return;
                }
                requestBody.gender = gender;
            }

            fetch(`/toggle_detection/${clientId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
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
        deactivateBtn.className = 'deactivate-btn';
        deactivateBtn.onclick = function () {
            const mode = modeSelect.value;
            if (!mode) {
                alert('Please select a detection mode to deactivate.');
                return;
            }

            let requestBody = {
                mode: mode,
                active: false
            };

            if (mode === 'gender') {
                requestBody.gender = 'None';
            }

            fetch(`/toggle_detection/${clientId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
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
        removeBtn.className = 'remove-btn';
        removeBtn.onclick = () => container.remove();

        controls.appendChild(toggleBtn);
        controls.appendChild(deactivateBtn);
        controls.appendChild(removeBtn);
        container.appendChild(controls);

        return container;
    }

    function updateStatusIndicators(clientId, statuses) {
        const status = statuses[clientId];
        if (!status) return;

        // Update motion status
        const motionStatus = document.getElementById(`motion-status-${clientId}`);
        if (motionStatus) {
            const indicator = motionStatus.querySelector('.indicator');
            indicator.textContent = status.motion_detected ? '🔴' : '⚪';
            indicator.style.color = status.motion_detected ? 'red' : 'gray';
        }

        // Update fight status
        const fightStatus = document.getElementById(`fight-status-${clientId}`);
        if (fightStatus) {
            const indicator = fightStatus.querySelector('.indicator');
            indicator.textContent = status.fight_detected ? '🔴' : '⚪';
            indicator.style.color = status.fight_detected ? 'red' : 'gray';
        }

        // Update gender status
        const genderStatus = document.getElementById(`gender-status-${clientId}`);
        if (genderStatus) {
            const indicator = genderStatus.querySelector('.indicator');
            if (status.mode === 'gender' && status.active) {
                indicator.textContent = status.detected_gender ? '🔴' : '🟢';
                indicator.style.color = status.detected_gender ? 'red' : 'green';
                genderStatus.innerHTML = `<span class="indicator">${indicator.textContent}</span> ${status.detected_gender || 'Monitoring'}`;
            } else {
                indicator.textContent = '⚪';
                indicator.style.color = 'gray';
                genderStatus.innerHTML = '<span class="indicator">⚪</span> Gender Detection';
            }
        }

        // Update UI to show active mode
        const modeSelect = document.getElementById(`mode-${clientId}`);
        const genderSelect = document.getElementById(`gender-${clientId}`);
        if (modeSelect && status.mode) {
            modeSelect.value = status.mode;
            if (status.mode === 'gender' && genderSelect) {
                genderSelect.style.display = 'inline-block';
                if (status.target_gender) {
                    genderSelect.value = status.target_gender;
                }
            }
        }
    }

    function checkForStreams() {
        fetch('/active_streams')
            .then(response => response.json())
            .then(data => {
                const activeClients = data.clients || [];
                const statuses = data.statuses || {};

                // Remove containers for clients that are no longer active
                document.querySelectorAll('.video-container').forEach(container => {
                    const clientId = container.id.replace('container-', '');
                    if (!activeClients.includes(clientId)) {
                        container.remove();
                    }
                });

                // Add or update containers for active clients
                activeClients.forEach(clientId => {
                    if (!document.getElementById(`container-${clientId}`)) {
                        videoGrid.appendChild(createVideoElement(clientId));
                    }

                    const img = document.getElementById(`video-${clientId}`);
                    if (img) {
                        // Refresh image source to prevent caching
                        img.src = `/video_feed/${encodeURIComponent(clientId)}?t=${Date.now()}`;
                    }

                    // Update status indicators
                    updateStatusIndicators(clientId, statuses);
                });
            })
            .catch(console.error);
    }

    checkForStreams();
    setInterval(checkForStreams, pollInterval);
});