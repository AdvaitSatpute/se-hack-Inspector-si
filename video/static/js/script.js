document.addEventListener('DOMContentLoaded', function() {
    const videoGrid = document.getElementById('video-grid');
    const maxStreams = 10; // Maximum number of streams to display
    const pollInterval = 2000; // Check for new streams every 2 seconds
    
    // Function to create a video element for a client
    function createVideoElement(clientId) {
        const container = document.createElement('div');
        container.className = 'video-container';
        
        const img = document.createElement('img');
        img.style.border = '2px solid red';  // Visual debug
        img.onload = () => console.log(`Stream ${clientId} loaded`);
        img.onerror = (e) => console.error(`Error ${clientId}:`, e);
        
        // Force fresh request each time
        img.src = `/video_feed/${encodeURIComponent(clientId)}?t=${Date.now()}`;
        
        container.appendChild(img);
        return container;
    }
    
    // Function to check for active streams
    function checkForStreams() {
        fetch('/active_streams')
            .then(response => response.json())
            .then(data => {
                const activeClients = data.clients || [];
                
                // Remove disconnected clients
                document.querySelectorAll('.video-container').forEach(container => {
                    const clientId = container.id.replace('container-', '');
                    if (!activeClients.includes(clientId)) {
                        container.remove();
                    }
                });
                
                // Add new clients
                activeClients.forEach(clientId => {
                    if (!document.getElementById(`container-${clientId}`) && videoGrid.children.length < maxStreams) {
                        videoGrid.appendChild(createVideoElement(clientId));
                    }
                });
                
                // Update video sources (to prevent caching/stale frames)
                activeClients.forEach(clientId => {
                    const videoElement = document.getElementById(`video-${clientId}`);
                    if (videoElement) {
                        videoElement.src = `/video_feed/${clientId}?t=${new Date().getTime()}`;
                    }
                });
            })
            .catch(error => console.error('Error fetching active streams:', error));
    }
    
    // Initial check
    checkForStreams();
    
    // Periodic checks
    setInterval(checkForStreams, pollInterval);
    
    // Handle window resize for responsive layout
    window.addEventListener('resize', function() {
        // The CSS grid handles most of this, but we can add additional logic if needed
    });
});