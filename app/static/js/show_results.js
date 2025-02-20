document.addEventListener("DOMContentLoaded", function () {
  const lamp = document.querySelector(".lamp");
  const container = document.querySelector(".container");
  const bulbContainer = document.getElementById("lampContainer");
  const loadingContainer = document.getElementById("loadingContainer");
  const circleProgress = document.querySelector(".circle-progress");
  const progressText = document.getElementById("progressText");
  const sleepingRobot = document.getElementById("sleepingRobot");
  const lampImage = document.getElementById("lampImage");

  let maxOffset = 339.292;
  let progress = 0;

  // Function to calculate the interval duration based on the prompt type
  function getProgressIntervalDuration(promptType) {
      if (promptType == 1) {
          return 50; // for 6 seconds, we need 60ms interval per progress step
      } else if (promptType == 2) {
        console.log(promptType)
          return 90; // for 15 seconds, we need 250ms interval per progress step
      }
      return 60; // Default case
  }

  // Adjust the progress bar animation based on the prompt type
  function animateProgressBar(promptType) {
      const intervalDuration = getProgressIntervalDuration(promptType);

      let loadingInterval = setInterval(() => {
          progress++;
          let offset = maxOffset - maxOffset * (progress / 100);
          circleProgress.style.strokeDashoffset = offset;
          progressText.textContent = progress + "%";
          
          if (progress >= 100) {
              clearInterval(loadingInterval);
              loadingContainer.classList.add("hidden");
              bulbContainer.style.display = "flex";
              sleepingRobot.style.display = "none"; 
              lampImage.style.display = "block";
          }
      }, intervalDuration); 
  }

  lamp.addEventListener("click", function () {
      if (lamp.classList.contains("on")) {
          lamp.classList.remove("on");
          container.classList.remove("visible");
          document.body.style.background = "#013220";
      } else {
          lamp.classList.add("on");
          container.classList.add("visible");
          document.body.style.background = "#f5f5dc";
      }
      if (lampImage.style.display === "none" || lampImage.style.display === "") {
          lampImage.style.display = "block";
      } else {
          lampImage.style.display = "none";
      }
  });

  // Function to send user data to API and handle results
  async function sendData(data) {
      const response = await fetch("/generate_response", {
          method: "POST",
          headers: {
              "Content-Type": "application/json",
          },
          body: JSON.stringify({
              prompt_type: localStorage.getItem('promptty'),
              profile: JSON.parse(localStorage.getItem('data'))
          }),
      });

      if (!response.ok) {
          throw new Error('Failed to send data!');
      }

      const responseData = await response.json();
      return responseData;
  }

  // Function to handle displaying the data in cards
  async function displayResults() {
      animateProgressBar(localStorage.getItem('promptty'));
      const result = await sendData();
      const promptType = parseInt(localStorage.getItem('promptty'));

      // Career Card remains the same
      const career = result.career || "No career found";
      const careerBack = document.querySelector('.card .card-back');
      careerBack.textContent = career;

      // Create step cards container
      const stepsContainer = document.createElement('div');
      stepsContainer.className = 'steps-container';
      cardsContainer.appendChild(stepsContainer);

      // Create step cards
      result.learning_path.forEach((step, index) => {
          const stepCard = document.createElement('div');
          stepCard.className = 'step-card';
          stepCard.style.display = index === 0 ? 'block' : 'none';

          const stepNumber = document.createElement('div');
          stepNumber.className = 'step-number';
          stepNumber.textContent = index + 1;

          const stepContent = document.createElement('div');
          stepContent.className = 'step-content';
          stepContent.innerHTML = promptType === 1
              ? `<h3>Step ${index + 1}</h3><p>${step}</p>`
              : `<h3></h3><p>${step}</p>`;

          // Create the "Back" button if it's not the first step
          if (index > 0) {
              const backButton = document.createElement('button');
              backButton.className = 'next-step';
              backButton.textContent = '← Back';
              backButton.addEventListener('click', () => {
                  stepCard.style.display = 'none';
                  stepsContainer.children[index - 1].style.display = 'block';
              });
              stepContent.appendChild(backButton);
          }

          // Create the "Next" button for all steps except the last
          if (index < result.learning_path.length - 1) {
              const nextButton = document.createElement('button');
              nextButton.className = 'next-step';
              nextButton.textContent = 'Next Step →';
              nextButton.addEventListener('click', () => {
                  stepCard.style.display = 'none';
                  stepsContainer.children[index + 1].style.display = 'block';
              });
              stepContent.appendChild(nextButton);
          }

          stepCard.appendChild(stepNumber);
          stepCard.appendChild(stepContent);
          stepsContainer.appendChild(stepCard);
      });
  }

  displayResults();
});
