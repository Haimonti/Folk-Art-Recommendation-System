// Check if the user is logged in
// if (!localStorage.getItem('userLoggedIn')) {
//     window.location.href = 'login.html';
// } else {
//     document.getElementById('usernameDisplay').textContent = localStorage.getItem('username');

const DOMAIN = window.CONFIG.DOMAIN;

    document.addEventListener("DOMContentLoaded", function () {
        const urlParams = new URLSearchParams(window.location.search);
        const userId = urlParams.get('userId');

        if (!userId) {
            alert('User ID not found. Please register first.');
            window.location.href = 'reg.html'; // Redirect to registration page
        }
        const container = document.getElementById('scrollContainer');
        let imageCount = 0;

        async function fetchHTML(url) {
            const response = await fetch(url);
            return response.text();
        }

        async function generateScrolls() {
            const html = await fetchHTML('scrolls/');
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const scrollDirs = [...doc.querySelectorAll('a')]
                .map(a => a.textContent)
                .filter(name => name.startsWith('s'));

            for (const [scrollIndex, scrollName] of scrollDirs.entries()) {
                const scrollDiv = document.createElement('div');
                scrollDiv.classList.add('scroll');

                // Scroll Collapsible Button
                const scrollHeader = document.createElement('button');
                scrollHeader.classList.add('collapsible');
                scrollHeader.textContent = `Scroll ${scrollIndex + 1}`;
                scrollDiv.appendChild(scrollHeader);

                // Scroll Content (Initially Hidden)
                const panelContainer = document.createElement('div');
                panelContainer.classList.add('panel-container');
                panelContainer.style.display = 'none';

                const imgHtml = await fetchHTML(`scrolls/${scrollName}/img/`);
                const imgDoc = parser.parseFromString(imgHtml, 'text/html');
                const images = [...imgDoc.querySelectorAll('a')]
                    .map(a => a.textContent)
                    .filter(name => name.endsWith('.jpg'));

                images.forEach((imgFile, panelIndex) => {
                    imageCount++;
                    const imageId = imageCount;
                    const panelWrapper = document.createElement('div');
                    panelWrapper.classList.add('panel-wrapper');

                    // Panel Collapsible Button
                    const panelHeader = document.createElement('button');
                    panelHeader.classList.add('panel-collapsible');
                    panelHeader.textContent = `Panel ${panelIndex + 1}`;
                    panelWrapper.appendChild(panelHeader);

                    // Panel Content (Initially Hidden)
                    const panelContent = document.createElement('div');
                    panelContent.classList.add('panel');
                    panelContent.style.display = 'none';

                    panelContent.innerHTML = `
                        <div class="section">
                            <img id="image-${imageId}" src="scrolls/${scrollName}/img/${imgFile}" alt="Frame Image" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto; object-fit: contain;">
                            <p style="text-align: center; margin-top: 10px;">Image ${imageId} in Panel ${panelIndex + 1} in Scroll ${scrollIndex + 1}</p>
                        </div>
                        <div class="section">
                            <label>Overall Rating:</label>
                            <div class="stars">
                                <input type="radio" id="star5-${scrollIndex}-${panelIndex}" name="rating-${scrollIndex}-${panelIndex}" value="5">
                                <label for="star5-${scrollIndex}-${panelIndex}">&#9733;</label>
                                <input type="radio" id="star4-${scrollIndex}-${panelIndex}" name="rating-${scrollIndex}-${panelIndex}" value="4">
                                <label for="star4-${scrollIndex}-${panelIndex}">&#9733;</label>
                                <input type="radio" id="star3-${scrollIndex}-${panelIndex}" name="rating-${scrollIndex}-${panelIndex}" value="3">
                                <label for="star3-${scrollIndex}-${panelIndex}">&#9733;</label>
                                <input type="radio" id="star2-${scrollIndex}-${panelIndex}" name="rating-${scrollIndex}-${panelIndex}" value="2">
                                <label for="star2-${scrollIndex}-${panelIndex}">&#9733;</label>
                                <input type="radio" id="star1-${scrollIndex}-${panelIndex}" name="rating-${scrollIndex}-${panelIndex}" value="1" required>
                                <label for="star1-${scrollIndex}-${panelIndex}">&#9733;</label>
                            </div>
                        </div>
                        <div class="section">
                            <label for="review-${scrollIndex}-${panelIndex}">Write Your Review:</label>
                            <textarea id="review-${scrollIndex}-${panelIndex}" name="review-${scrollIndex}-${panelIndex}" rows="4" placeholder="What did you like or dislike?" required></textarea>
                        </div>
                        <div class="section">
                            <button type="submit" class="submit-button" data-scroll-id="${scrollIndex + 1}" data-panel-id="${panelIndex + 1}" data-image-id="${imageId}" data-user-id="${userId}">Submit</button>
                        </div>
                    `;

                    panelWrapper.appendChild(panelContent);
                    panelContainer.appendChild(panelWrapper);
                    imageCount++;
                });

                scrollDiv.appendChild(panelContainer);
                container.appendChild(scrollDiv);
            }

        // Attach event listeners after elements are added
        addCollapsibleListeners();
        addSubmitListeners();
    }

        function addCollapsibleListeners() {
            document.querySelectorAll('.collapsible').forEach(button => {
                button.addEventListener('click', function () {
                    this.nextElementSibling.style.display =
                        this.nextElementSibling.style.display === 'none' ? 'block' : 'none';
                });
            });

            document.querySelectorAll('.panel-collapsible').forEach(button => {
                button.addEventListener('click', function () {
                    this.nextElementSibling.style.display =
                        this.nextElementSibling.style.display === 'none' ? 'block' : 'none';
                });
            });
        }

        function addSubmitListeners() {
            document.querySelectorAll('.submit-button').forEach(button => {
                button.addEventListener('click', async function () {
                    const scrollId = parseInt(this.getAttribute("data-scroll-id")) || 0;
                    console.log('SrollId: ', scrollId);
                    const panelId = parseInt(this.getAttribute("data-panel-id")) || 0;
                    console.log('PanelId: ', panelId);
                    const imageId = parseInt(event.target.getAttribute("data-image-id")) || 0;
                    console.log("Image ID:", imageId);
                    const userId = parseInt(this.getAttribute("data-user-id")) || 0;
                    console.log("User ID:", userId);


                    if (!scrollId || !panelId || !imageId) {
                        console.error("Missing attributes on submit button.");
                        return;
                    }

                    const phaseId = 1; // TODO: Replace with actual phase ID

                    // Get rating
                    const ratingInput = document.querySelector(`input[name="rating-${scrollId - 1}-${panelId - 1}"]:checked`);
                    const rating = ratingInput ? parseInt(ratingInput.value) : null;

                    // Get review text
                    const reviewTextArea = document.getElementById(`review-${scrollId - 1}-${panelId - 1}`);
                    const review = reviewTextArea.value;

                    if (!rating) {
                        alert("Please provide a rating");
                        return;
                    }

                    // Prepare request payload
                    const requestData = {
                        userId,
                        phaseId,
                        scrollId,
                        panelId,
                        imageId,
                        rating,
                        review
                    };

                    const url = `${DOMAIN}/userScrollPreferences`;

                    try {
                        const response = await fetch(url, {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify(requestData)
                        });

                        if (response.ok) {
                            // Disable submit button
                            this.disabled = true;
                            this.style.backgroundColor = "#ccc";
                            this.style.cursor = "not-allowed";

                            // Disable rating inputs
                            document.querySelectorAll(`input[name="rating-${scrollId - 1}-${panelId - 1}"]`).forEach(input => {
                                input.disabled = true;
                            });

                            // Make review text read-only and greyed out
                            reviewTextArea.readOnly = true;
                            reviewTextArea.style.backgroundColor = "#f0f0f0";
                            reviewTextArea.style.color = "#666";

                        } else {
                            alert("Error submitting review. Please try again.");
                        }
                    } catch (error) {
                        console.error("Error submitting review:", error);
                        alert("Failed to submit review due to a network error.");
                    }
                });
            });
        }

        generateScrolls();
    });
// }
