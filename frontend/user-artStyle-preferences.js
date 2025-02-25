const DOMAIN = window.CONFIG.DOMAIN;

document.addEventListener('DOMContentLoaded', function () {
    // Extract userId from the URL query parameters
    const urlParams = new URLSearchParams(window.location.search);
    const userId = urlParams.get('userId');

    if (!userId) {
        alert('User ID not found. Please register first.');
        window.location.href = 'reg.html'; // Redirect to registration page
    }

    document.getElementById('preferenceForm').addEventListener('submit', async function (event) {
        event.preventDefault();

        const formData = new FormData(event.target);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Add userId to the form data
        data.userId = userId;
        const url = `${DOMAIN}/userArtPreferences`;

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (response.ok) {
                const result = await response.json();
                alert('Preferences saved successfully: ' + result.message);
                window.location.replace(`user-scroll-preferences.html?userId=${result.userId}`);
            } else {
                const error = await response.json();
                alert('Error: ' + (error.error || 'Something went wrong'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error: Could not connect to the server.');
        }
    });
});