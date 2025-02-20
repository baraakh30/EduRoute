const chatBody = document.querySelector(".chat-body");
const messageInput = document.querySelector(".message-input");
const sendMessage = document.querySelector("#send-message");

const chatbotToggler = document.querySelector("#chatbot-toggler");
const closeChatbot = document.querySelector("#close-chatbot");

// API setup


// Initialize user message and file data
const userData = {
    message: null,
    file: {
        data: null,
        mime_type: null,
    },
};

// Store chat history
const chatHistory = [];
const initialInputHeight = messageInput.scrollHeight;

// Create message element with dynamic classes and return it
const createMessageElement = (content, ...classes) => {
    const div = document.createElement("div");
    div.classList.add("message", ...classes);
    div.innerHTML = content;
    return div;
};

// Generate bot response using API
const generateBotResponse = async (incomingMessageDiv) => {
    const messageElement = incomingMessageDiv.querySelector(".message-text");

    // Add user message to chat history
    chatHistory.push({
        role: "user",
        parts: [{ text: userData.message }, ...(userData.file.data ? [{ inline_data: userData.file }] : [])],
    });

    // API request options
    const requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            prompt: chatHistory,
        }),
    };

    try {
        // Fetch bot response from API and handle the response stream
        const response = await fetch('/chatbot', requestOptions);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let text = '';

        while (!done) {
            // Inside your while loop:
            const { value, done: doneReading } = await reader.read();
            done = doneReading;

            // Add this replacement filter
            const chunkText = decoder.decode(value, { stream: true })
                                                .replace(/<\|im_end\|>/g, ''); // Remove special token

            text += chunkText;

            // Update the message text as the response comes in
            messageElement.innerText = text.replace(/<\|im_end\|>/g, ''); // Double check replacement
            chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
        }

        // Add the bot response to chat history once it's fully received
        chatHistory.push({
            role: "model",
            parts: [{ text }],
        });

    } catch (error) {
        // Handle error in API response
        console.log(error);
        messageElement.innerText = error.message;
        messageElement.style.color = "#ff0000";
    } finally {
        // Reset user's file data, remove thinking indicator, and scroll chat to the bottom
        userData.file = {};
        incomingMessageDiv.classList.remove("thinking");
        chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
    }
};


// Handle outgoing user messages
const handleOutgoingMessage = (e) => {
    e.preventDefault();
    userData.message = messageInput.value.trim();
    messageInput.value = "";
    messageInput.dispatchEvent(new Event("input"));

    // Create and display user message
    const messageContent = `<div class="message-text"></div>
                                 ${userData.file.data ? `<img src="data:${userData.file.mime_type};base64,${userData.file.data}" class="attachment" />` : ""}`;

    const outgoingMessageDiv = createMessageElement(messageContent, "user-message");
    outgoingMessageDiv.querySelector(".message-text").innerText = userData.message;
    chatBody.appendChild(outgoingMessageDiv);
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

    // Simulate bot response with thinking indicator after a delay
    setTimeout(() => {
        const messageContent = `<span class="material-symbols-rounded bot-avatar-icon">android</span>
                                 <div class="message-text">
                                     <div class="thinking-indicator">
                                         <div class="dot"></div>
                                         <div class="dot"></div>
                                         <div class="dot"></div>
                                     </div>
                                 </div>`;

        const incomingMessageDiv = createMessageElement(messageContent, "bot-message", "thinking");
        chatBody.appendChild(incomingMessageDiv);
        chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
        generateBotResponse(incomingMessageDiv);
    }, 600);
};

// Adjust input field height dynamically
messageInput.addEventListener("input", () => {
    messageInput.style.height = `${initialInputHeight}px`;
    messageInput.style.height = `${messageInput.scrollHeight}px`;
    document.querySelector(".chat-form").style.borderRadius = messageInput.scrollHeight > initialInputHeight ? "15px" : "32px";
});

// Handle Enter key press for sending messages
messageInput.addEventListener("keydown", (e) => {
    const userMessage = e.target.value.trim();
    if (e.key === "Enter" && !e.shiftKey && userMessage && window.innerWidth > 768) {
        handleOutgoingMessage(e);
    }
});


document.querySelector(".chat-form").appendChild(picker);
messageInput.addEventListener("input", () => {
    messageInput.style.height = `${initialInputHeight}px`;
    messageInput.style.height = `${messageInput.scrollHeight}px`;
    document.querySelector(".chat-form").style.borderRadius = messageInput.scrollHeight > initialInputHeight ? "15px" : "32px";
  });
  
sendMessage.addEventListener("click", (e) => handleOutgoingMessage(e));

// script_chat.js

    const chatbotPopup = document.querySelector('.chatbot-popup'); // If you decide to re-implement the popup
    const sendButton = document.getElementById('send-message');
    const chatForm = document.getElementById('chat-form');



    // --- Emoji Picker (if you want to include it) ---
    const emojiPickerButton = document.getElementById('emoji-picker');
    const emojiPicker = new EmojiMart.Picker({     onEmojiSelect: (emoji) => {
        const { selectionStart: start, selectionEnd: end } = messageInput;
        messageInput.setRangeText(emoji.native, start, end, "end");
        messageInput.focus();
    },  set: 'apple' }); // You can customize emoji set
    document.body.appendChild(emojiPicker);
    emojiPicker.classList.add('chat-emoji-picker'); // Add a class for CSS targeting
    emojiPicker.style.display = 'none'; // Initially hide the picker
    emojiPickerButton.addEventListener("click", (event) => {
        event.preventDefault();
        console.log("Emoji Picker Clicked!"); // Debugging log
        emojiPicker.classList.toggle("active");
    });
    

    emojiPickerButton.addEventListener("click", (event) => {
        event.preventDefault();
        emojiPicker.style.display = emojiPicker.style.display === "none" ? "block" : "none";
    });
    
    function appendEmoji(emoji) {
        messageInput.value += emoji.native;
        emojiPicker.style.display = 'none'; // Hide picker after selection
    }
    // --- End Emoji Picker ---



    // --- Handle Sending Messages ---
    chatForm.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent default form submission

        const messageText = messageInput.value.trim();
        if (!messageText) return;

        // 1. Display User Message in Chat Log
        displayUserMessage(messageText);
        messageInput.value = ""; // Clear input after sending

        // 2. Simulate Bot Response (Replace with actual AI interaction)
        simulateBotResponse(messageText); // Pass user message to bot response function

    });


    function displayUserMessage(message) {
        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('message', 'user-message');
        userMessageDiv.innerHTML = `
            <div class="message-text">${message}</div>
            <img class="user-avatar" src="user_avatar.png" alt="User Avatar">
        `; // You can set user avatar image here or dynamically

        chatBody.appendChild(userMessageDiv);
        chatBody.scrollTop = chatBody.scrollHeight; // Scroll to bottom
    }


    function simulateBotResponse(userMessage) {
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('message', 'bot-message', 'thinking'); // Add 'thinking' class initially
        botMessageDiv.innerHTML = `
            <span class="material-symbols-rounded bot-avatar-icon">android</span>
            <div class="message-text">
                <div class="thinking-indicator">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
            </div>
        `;
        chatBody.appendChild(botMessageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;


        // Simulate delay for bot response (replace with actual AI call)
        setTimeout(() => {
            botMessageDiv.classList.remove('thinking'); // Remove thinking indicator

            const responseText = getBotResponse(userMessage); // Replace with your AI logic
            botMessageDiv.querySelector('.message-text').innerHTML = responseText;
            chatBody.scrollTop = chatBody.scrollHeight; // Scroll to bottom after response
        }, 1500); // Simulate 1.5 seconds thinking time
    }


    function getBotResponse(userMessage) {
        // --- Replace this with your actual AI interaction logic ---
        // For now, a simple example response:
        return `Echo: "${userMessage}".  I'm an AI expert ready to help you with your educational path!`;
    }


