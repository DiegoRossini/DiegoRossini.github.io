document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded and parsed');

    const form = document.querySelector('form');

    if (form) {
        form.addEventListener('submit', (event) => {
            event.preventDefault();

            console.log('Form submit event triggered');

            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const message = document.getElementById('message').value;

            console.log('Form data:', { name, email, message });

            // Replace this URL with your Ngrok URL
            const ngrokUrl = 'https://6b80-2a01-e0a-e10-bb10-5d01-c632-231-8404.ngrok-free.app';

            fetch(`${ngrokUrl}/send-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name, email, message })
            })
            .then(response => response.text())
            .then(data => {
                console.log('Server response:', data);
                alert('Email sent successfully');
                form.reset(); // Reset the form fields after successful submission
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to send email');
            });
        });
    } else {
        console.error('Form element not found');
    }
});
