import './chatwindow.css';

async function handleSubmit(event) {
    event.preventDefault();
    const text = event.target.text.value;
    getSentiment(text)
        .then(sentiment => alert(sentiment));
}

async function getSentiment(text) {
    const sentimentRequest = {text: text};
    const fetchRequest = {
        method: 'POST',
        mode: 'cors',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(sentimentRequest)
    };
    const response = await fetch('http://127.0.0.1:8000/predict', fetchRequest);
    const json = await response.json();
    return json.sentiment;
}

export default function ChatWindow() {
    return (
        <div id="chatWindow">
            <form onSubmit={handleSubmit}>
                <input type="text" id="text" name="text" />
                <input type="submit" value="Submit"/>
                {/* <input type="text" id="chatInput" placeholder="Type a message..." />
                <input type="button"> Send </input> */}
            </form>
        </div>
    );
}