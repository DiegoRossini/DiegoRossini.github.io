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
            
            fetch('http://localhost:3000/send-email', {
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
