import {CONFIG} from "./config";

const DOMAIN = CONFIG.DOMAIN;

async function submitForm(event) {
    event.preventDefault();

    const form = event.target;

    const formData = new FormData(form);

    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    const url = `${DOMAIN}/users`;

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
            alert('Registration successful: ' + result.message);
            form.reset();
            window.location.replace(`user-artStyle-preferences.html?userId=${result.userId}`);
        } else {
            const error = await response.json();
            alert('Error: ' + (error.error || 'Something went wrong'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error: Could not connect to the server.');
    }
}

// Need a landing Page where need to show 2 Phases

// id(unique & primary Id) | userId | phaseId | scrollId | panelId | imageId | rating | review |