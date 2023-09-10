const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const toggleSwitch = document.querySelector('.toggle-switch');
const modeText = document.getElementById('mode-text');

// Function to display a message in the chat window
function displayMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'user-message' : 'bot-message';
    messageDiv.textContent = message;
    chatWindow.appendChild(messageDiv);
}

// Event listener for sending messages
sendButton.addEventListener('click', () => {
    const userMessage = userInput.value;
    displayMessage(userMessage, true); // Display the user's message in the chat window
    userInput.value = ''; // Clear the input field

    // Send the user's message to the backend (implement this part as previously shown)
});

// Event listener for dark mode toggle
toggleSwitch.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    if (document.body.classList.contains('dark-mode')) {
        modeText.textContent = 'On';
    } else {
        modeText.textContent = 'Off';
    }
});
