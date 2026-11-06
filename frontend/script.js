const API_URL = "http://127.0.0.1:8000/chat";
const conversationId = crypto.randomUUID();


const userIdInput =
    document.getElementById("userId");

const messageInput =
    document.getElementById("messageInput");

const sendButton =
    document.getElementById("sendButton");

const chatMessages =
    document.getElementById("chatMessages");

const resultInfo =
    document.getElementById("resultInfo");

const confidenceValue =
    document.getElementById("confidenceValue");

const sourceInfo =
    document.getElementById("sourceInfo");

const sourceValue =
    document.getElementById("sourceValue");

const escalationInfo =
    document.getElementById("escalationInfo");

const ticketInfo =
    document.getElementById("ticketInfo");

const ticketId =
    document.getElementById("ticketId");


/* =========================
   ADD MESSAGE
========================= */

function addMessage(sender, message) {

    const messageDiv =
        document.createElement("div");

    messageDiv.classList.add(
        "message",
        sender
    );


    const avatar =
        document.createElement("div");

    avatar.classList.add(
        "avatar"
    );


    if (sender === "assistant") {

        avatar.classList.add(
            "assistant-avatar"
        );

        avatar.textContent = "✦";

    } else {

        avatar.classList.add(
            "user-avatar"
        );

        avatar.textContent = "👤";
    }


    const content =
        document.createElement("div");

    content.classList.add(
        "message-content"
    );


    const label =
        document.createElement("div");

    label.classList.add(
        "message-label"
    );


    const senderName =
        document.createElement("span");

    senderName.textContent =
        "Now";


    label.textContent =
        sender === "user"
            ? "You"
            : "SupportAI";


    label.appendChild(senderName);


    const bubble =
        document.createElement("div");

    bubble.classList.add(
        "message-bubble"
    );


    /*
    * Render Markdown formatting from
    * the AI response.
    */
    if (sender === "assistant") {
        const markdown = message.replace(/\\\*\*/g, "**");
        bubble.innerHTML = marked.parse(markdown);
    } else {
        bubble.textContent = message;
    }


    content.appendChild(label);

    content.appendChild(bubble);


    messageDiv.appendChild(avatar);

    messageDiv.appendChild(content);


    chatMessages.appendChild(
        messageDiv
    );


    scrollToBottom();
}


/* =========================
   LOADING MESSAGE
========================= */

function addLoadingMessage() {

    const messageDiv =
        document.createElement("div");

    messageDiv.id =
        "loadingMessage";

    messageDiv.classList.add(
        "message",
        "assistant"
    );


    const avatar =
        document.createElement("div");

    avatar.classList.add(
        "avatar",
        "assistant-avatar"
    );

    avatar.textContent = "✦";


    const content =
        document.createElement("div");

    content.classList.add(
        "message-content"
    );


    const label =
        document.createElement("div");

    label.classList.add(
        "message-label"
    );

    label.textContent =
        "SupportAI";


    const bubble =
        document.createElement("div");

    bubble.classList.add(
        "message-bubble"
    );

    bubble.textContent =
        "Thinking...";


    content.appendChild(label);

    content.appendChild(bubble);

    messageDiv.appendChild(avatar);

    messageDiv.appendChild(content);


    chatMessages.appendChild(
        messageDiv
    );


    scrollToBottom();
}


/* =========================
   REMOVE LOADING
========================= */

function removeLoadingMessage() {

    const loading =
        document.getElementById(
            "loadingMessage"
        );

    if (loading) {

        loading.remove();
    }
}


/* =========================
   SCROLL
========================= */

function scrollToBottom() {

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


/* =========================
   RESULT INFORMATION
========================= */

function updateResult(response) {

    resultInfo.classList.remove(
        "hidden"
    );


    const confidence =
        Math.round(
            (response.confidence || 0) * 100
        );


    confidenceValue.textContent =
        `${confidence}%`;


    sourceInfo.classList.add(
        "hidden"
    );

    escalationInfo.classList.add(
        "hidden"
    );

    ticketInfo.classList.add(
        "hidden"
    );


    /*
     * Confidence >= 85 generally means
     * the answer has strong supporting
     * context.
     */
    if (confidence >= 85) {

        sourceInfo.classList.remove(
            "hidden"
        );

        sourceValue.textContent =
            "Verified context";
    }


    if (response.escalated) {

        escalationInfo.classList.remove(
            "hidden"
        );
    }


    if (response.ticket_id) {

        ticketInfo.classList.remove(
            "hidden"
        );

        ticketId.textContent =
            response.ticket_id;
    }
}


/* =========================
   SEND MESSAGE
========================= */

async function sendMessage() {

    const userId =
        userIdInput.value.trim();

    const message =
        messageInput.value.trim();


    if (!userId) {

        alert(
            "Please enter your Customer ID."
        );

        userIdInput.focus();

        return;
    }


    if (!message) {

        messageInput.focus();

        return;
    }


    /*
     * Show user's message.
     */
    addMessage(
        "user",
        message
    );


    messageInput.value = "";


    sendButton.disabled = true;

    sendButton.textContent =
        "•••";


    addLoadingMessage();


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        user_id:Number(userId),

                        message:message,

                        conversation_id:conversationId
                    })
                }
            );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data =
            await response.json();


        removeLoadingMessage();


        /*
         * Show AI response.
         */
        addMessage(
            "assistant",
            data.answer ||
            "No response was generated."
        );


        updateResult(data);


    } catch (error) {

        console.error(error);


        removeLoadingMessage();


        addMessage(
            "assistant",
            "I couldn't connect to the support server. Please make sure FastAPI is running."
        );


    } finally {

        sendButton.disabled = false;

        sendButton.textContent =
            "➤";

        messageInput.focus();
    }
}


/* =========================
   QUICK ACTIONS
========================= */

document
    .querySelectorAll(".quick-action")
    .forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const message =
                    button.dataset.message;

                messageInput.value =
                    message;

                messageInput.focus();
            }
        );

    });


/* =========================
   EVENTS
========================= */

sendButton.addEventListener(
    "click",
    sendMessage
);


messageInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }

    }
);