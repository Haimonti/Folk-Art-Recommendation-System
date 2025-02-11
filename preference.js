// Check if the user is logged in
// if (!localStorage.getItem('userLoggedIn')) {
//     window.location.href = 'login.html';
// } else {
//     document.getElementById('usernameDisplay').textContent = localStorage.getItem('username');

    document.addEventListener("DOMContentLoaded", function () {
        const container = document.getElementById('scrollContainer');

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

            for (const scrollName of scrollDirs) {
                const scrollDiv = document.createElement('div');
                scrollDiv.classList.add('scroll');

                // Scroll Collapsible Button
                const scrollHeader = document.createElement('button');
                scrollHeader.classList.add('collapsible');
                scrollHeader.textContent = scrollName;
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

                images.forEach((imgFile, index) => {
                    const panelWrapper = document.createElement('div');
                    panelWrapper.classList.add('panel-wrapper');

                    // Panel Collapsible Button
                    const panelHeader = document.createElement('button');
                    panelHeader.classList.add('panel-collapsible');
                    panelHeader.textContent = `Panel ${index + 1}`;
                    panelWrapper.appendChild(panelHeader);

                    // Panel Content (Initially Hidden)
                    const panelContent = document.createElement('div');
                    panelContent.classList.add('panel');
                    panelContent.style.display = 'none';

                    panelContent.innerHTML = `
                    <div class="section">
                        <img src="scrolls/${scrollName}/img/${imgFile}" alt="Frame Image" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto; object-fit: contain;">
                        <p style="text-align: center; margin-top: 10px;">Description for Panel ${index + 1} in Scroll ${scrollName}</p>
                    </div>
                    <div class="section">
                        <label>Overall Rating:</label>
                        <div class="stars">
                            <input type="radio" id="star5-${scrollName}-${index}" name="rating-${index}" value="5">
                            <label for="star5-${scrollName}-${index}">&#9733;</label>
                            <input type="radio" id="star4-${scrollName}-${index}" name="rating-${index}" value="4">
                            <label for="star4-${scrollName}-${index}">&#9733;</label>
                            <input type="radio" id="star3-${scrollName}-${index}" name="rating-${index}" value="3">
                            <label for="star3-${scrollName}-${index}">&#9733;</label>
                            <input type="radio" id="star2-${scrollName}-${index}" name="rating-${index}" value="2">
                            <label for="star2-${scrollName}-${index}">&#9733;</label>
                            <input type="radio" id="star1-${scrollName}-${index}" name="rating-${index}" value="1" required>
                            <label for="star1-${scrollName}-${index}">&#9733;</label>
                        </div>
                    </div>
                    <div class="section">
                        <label for="review-${scrollName}-${index}">Write Your Review:</label>
                        <textarea id="review-${scrollName}-${index}" name="review-${index}" rows="4" placeholder="What did you like or dislike?" required></textarea>
                    </div>
                    <div class="section">
                        <button type="submit" id="submit-${scrollName}-${index}">Submit</button>
                    </div>
                `;

                    panelWrapper.appendChild(panelContent);
                    panelContainer.appendChild(panelWrapper);
                });

                scrollDiv.appendChild(panelContainer);
                container.appendChild(scrollDiv);
            }

            // Attach event listeners after elements are added
            addCollapsibleListeners();
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

        generateScrolls();
    });
// }
