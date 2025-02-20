export let promptty = 1; // Default value is 1 (short answer)
export let data = {};

document.addEventListener('DOMContentLoaded', () => {
  const nextBtns = document.querySelectorAll('.next-btn');
  const backBtns = document.querySelectorAll('.back-btn');
  const questions = document.querySelectorAll('.question');
  const answers = document.querySelectorAll('.choice-btn');
  const name = document.getElementById('name');
  const startBtn = document.getElementById('start');
  const errorMessage = document.getElementById('error');
  const customInterestInput = document.querySelector('.custom-interest-input');
  const addInterestBtn = document.querySelector('.add-interest-btn');
  const addedInterestsContainer = document.querySelector('.added-interests');
  const customKnowledgeInput = document.querySelector('.custom-knowledge-input');
  const addKnowledgeBtn = document.querySelector('.add-knowledge-btn');
  const addedKnowledgeContainer = document.querySelector('.added-knowledge');
   const submitBtn = document.getElementById('submit');

  let count = 0;


  // Handle short/long answer selection
  const shortAnswerBtn = document.getElementById('short-answer');
  const longAnswerBtn = document.getElementById('long-answer');
  const warningMessage = document.getElementById('warning-message');

  shortAnswerBtn.addEventListener('click', () => {
    delete data["prompt"];

    promptty = 1;
    warningMessage.style.display = 'none'; // Hide warning if short answer
  });

  longAnswerBtn.addEventListener('click', () => {
    delete data["prompt"];
    promptty = 2;
    warningMessage.style.display = 'block'; // Show warning if long answer
  });

  // Function to display the current question
  function displayQuestion () {
    questions.forEach((question, index) => {
      question.classList.toggle('active', index === count);
    });
  }

  // Disable nextBtn until user makes a choice
  if(count > 0 && count < nextBtns.length) {
    nextBtns[count].disabled = true;
  }

  // Handle name input and enable the start button
  name.addEventListener('input', () => {
    if (!name.value.trim()) {
      errorMessage.style.display = 'block';
      startBtn.disabled = true;
    } else {
      errorMessage.style.display = 'none';
      startBtn.disabled = false;
    }
  });

  // Start assessment and store the name
  startBtn.addEventListener('click', () => {
    if (!name.value.trim()) {
      errorMessage.style.display = 'block';
    } 
  });

  // Handle user's choices for multiple and single selection questions
  answers.forEach(answer => {
    answer.addEventListener('click', (e) => {
      const questionId = questions[count].id;
      const selectedBtn = e.target;
      const parent = selectedBtn.parentElement;

      if (questionId === 'interests' || questionId === 'previous_knowledge') {
        if (!data[questionId]) {
          data[questionId] = [];
        }

        if (selectedBtn.classList.contains('selected')) {
          selectedBtn.classList.remove('selected');
          data[questionId] = data[questionId].filter(choice => choice !== selectedBtn.innerText);
        } else {
          selectedBtn.classList.add('selected');
          data[questionId].push(selectedBtn.innerText);
        }

        if (data[questionId].length === 0) {
          delete data[questionId];
        }
      } else {
        // Handling single selection for other questions
        if (selectedBtn.classList.contains('selected')) {
          selectedBtn.classList.remove('selected');
          delete data[questionId];
        } else {
          parent.querySelectorAll('.choice-btn').forEach(btn => btn.classList.remove('selected'));
          selectedBtn.classList.add('selected');
          data[questionId] = selectedBtn.innerText;
        }
      }
      if (data["prompt"])
      delete data["prompt"];

      // Enable nextBtn if a choice is made
      if(nextBtns[count])
        if (parent.querySelector('.selected')) {
          nextBtns[count].disabled = false;
        } else {
          nextBtns[count].disabled = true;
        }
        console.log(JSON.stringify(data))
    });
  });

  // Add custom interest input
  addInterestBtn.addEventListener('click', () => {
    const customInterest = customInterestInput.value.trim();
    if (customInterest) {
      const button = document.createElement('button');
      button.type = 'button';
      button.classList.add('choice-btn', 'p-3');
      button.textContent = customInterest;

      // Make custom button selectable by default (add 'selected' class)
      button.classList.add('selected');
      if (!data['interests']) {
        data['interests'] = [];
      }
      data['interests'].push(customInterest);

      // Add delete button for custom interest
      const deleteBtn = document.createElement('span');
      deleteBtn.classList.add('delete-btn');
      deleteBtn.textContent = '✖';
      deleteBtn.style.marginLeft = '8px'; // Move delete button to the right
      deleteBtn.addEventListener('click', () => {
        button.remove();
        data['interests'] = data['interests'].filter(item => item !== customInterest);
      });

      button.appendChild(deleteBtn);
      addedInterestsContainer.appendChild(button);
      customInterestInput.value = ''; // Clear the input
    }
  });

  // Add custom knowledge input for previous_knowledge question
  addKnowledgeBtn.addEventListener('click', () => {
    const customKnowledge = customKnowledgeInput.value.trim();
    if (customKnowledge) {
      const button = document.createElement('button');
      button.type = 'button';
      button.classList.add('choice-btn', 'p-3');
      button.textContent = customKnowledge;

      // Make custom button selectable by default (add 'selected' class)
      button.classList.add('selected');
      if (!data['previous_knowledge']) {
        data['previous_knowledge'] = [];
      }
      data['previous_knowledge'].push(customKnowledge);

      // Add delete button for custom knowledge
      const deleteBtn = document.createElement('span');
      deleteBtn.classList.add('delete-btn');
      deleteBtn.textContent = '✖';
      deleteBtn.style.marginLeft = '8px'; // Move delete button to the right
      deleteBtn.addEventListener('click', () => {
        button.remove();
        data['previous_knowledge'] = data['previous_knowledge'].filter(item => item !== customKnowledge);
      });

      button.appendChild(deleteBtn);
      addedKnowledgeContainer.appendChild(button);
      customKnowledgeInput.value = ''; // Clear the input
    }
  });

  // Event listener for the next button
  nextBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (count < questions.length - 1) {
        count++;
        displayQuestion();
      }
    });
  });

  // Event listener for the back button
  backBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (count > 0) {
        count--;
        displayQuestion();
      }
    });
  });

  submitBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
        localStorage.setItem('promptty', promptty);
        localStorage.setItem('data', JSON.stringify(data)); // Convert object to string
     
      window.location.href = '/results';  // Redirect to the /results page
  
    } catch (error) {
      console.log('Error sending data:', error);
    }
  });

  displayQuestion();
});
