// Check if the user is logged in
if (!localStorage.getItem('userLoggedIn')) {
    window.location.href = 'login.html';
} else {
    document.getElementById('usernameDisplay').textContent = localStorage.getItem('username');

    const numScrolls = 3; // Number of scrolls
    const numPanels = 5; // Number of panels per scroll

    function generateScrolls() {
        const container = document.getElementById('scrollContainer');

        for (let scrollIndex = 0; scrollIndex < numScrolls; scrollIndex++) {
            const scroll = document.createElement('div');
            scroll.classList.add('scroll');

            // Create collapsible button for the scroll
            const scrollHeader = document.createElement('button');
            scrollHeader.classList.add('collapsible');
            scrollHeader.textContent = `Scroll ${scrollIndex + 1}`;
            scroll.appendChild(scrollHeader);

            // Scroll Content (Initially Hidden)
            const panelContainer = document.createElement('div');
            panelContainer.classList.add('panel-container');
            panelContainer.style.display = 'none';

            for (let panelIndex = 0; panelIndex < numPanels; panelIndex++) {
                const panelWrapper = document.createElement('div');
                panelWrapper.classList.add('panel-wrapper');

                // Create collapsible button for each panel
                const panelHeader = document.createElement('button');
                panelHeader.classList.add('panel-collapsible');
                panelHeader.textContent = `Panel ${panelIndex + 1}`;
                panelWrapper.appendChild(panelHeader);

                // Panel Content (Initially Hidden)
                const panel = document.createElement('div');
                panel.classList.add('panel');
                panel.style.display = 'none';

                panel.innerHTML = `
          <div class="section">
            <img src="scrolls/s1/img/s${scrollIndex + 1}.jpg" alt="Frame Image" style="display: block; margin: auto;" height="100" width="100">
            <p style="text-align: center; margin-top: 10px;">Description for Panel ${panelIndex + 1} in Scroll ${scrollIndex + 1}</p>
          </div>
          <div class="section">
            <label>Overall Rating:</label>
            <div class="stars">
              <input type="radio" id="star5-${scrollIndex}-${panelIndex}" name="rating-${panelIndex}" value="5"><label for="star5-${scrollIndex}-${panelIndex}">&#9733;</label>
              <input type="radio" id="star4-${scrollIndex}-${panelIndex}" name="rating-${panelIndex}" value="4"><label for="star4-${scrollIndex}-${panelIndex}">&#9733;</label>
              <input type="radio" id="star3-${scrollIndex}-${panelIndex}" name="rating-${panelIndex}" value="3"><label for="star3-${scrollIndex}-${panelIndex}">&#9733;</label>
              <input type="radio" id="star2-${scrollIndex}-${panelIndex}" name="rating-${panelIndex}" value="2"><label for="star2-${scrollIndex}-${panelIndex}">&#9733;</label>
              <input type="radio" id="star1-${scrollIndex}-${panelIndex}" name="rating-${panelIndex}" value="1" required><label for="star1-${scrollIndex}-${panelIndex}">&#9733;</label>
            </div>
          </div>
          <div class="section">
            <label for="review-${scrollIndex}-${panelIndex}">Write Your Review:</label>
            <textarea id="review-${scrollIndex}-${panelIndex}" name="review-${panelIndex}" rows="4" style="width: 100%;" placeholder="What did you like or dislike?" required></textarea>
          </div>
          <div class="section">
            <button type="submit" id="submit-${scrollIndex}-${panelIndex}">Submit</button>
          </div>
        `;

                panelWrapper.appendChild(panel);
                panelContainer.appendChild(panelWrapper);

                // Toggle visibility for panels inside the scroll
                panelHeader.addEventListener('click', function () {
                    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
                });
            }

            scroll.appendChild(panelContainer);

            // Toggle visibility for entire scroll
            scrollHeader.addEventListener('click', function () {
                panelContainer.style.display = panelContainer.style.display === 'none' ? 'block' : 'none';
            });

            container.appendChild(scroll);
        }
    }

    window.onload = generateScrolls;
}
