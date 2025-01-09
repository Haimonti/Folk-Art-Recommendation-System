const loginForm = document.getElementById("login-form");
const loginButton = document.getElementById("login-form-submit");
const loginErrorMsg = document.getElementById("login-error-msg");

// When the login button is clicked, the following code is executed
loginButton.addEventListener("click", async (e) => {
    // Prevent the default submission of the form
    e.preventDefault();
    // Get the values input by the user in the form fields
    const username = loginForm.username.value;
    const password = loginForm.password.value;

    try {
        const data = {};
        data['username'] = username;
        data['password'] = password;
        const response = await fetch('http://127.0.0.1:5000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (response.ok) {
            const result = await response.json();
            alert('Registration successful: ' + result.message);
        } else {
            const error = await response.json();
            // Otherwise, make the login error message show (change its oppacity)
            loginErrorMsg.style.opacity = 1;
            alert('Error: ' + (error.error || 'Something went wrong'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error: Could not connect to the server.');
    }
})