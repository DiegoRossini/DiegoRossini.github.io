function copyEmail() {
    const email = 'diego.rossini1993@gmail.com';
    navigator.clipboard.writeText(email).then(function() {
        alert('Email address copied to clipboard!');
    }, function(err) {
        console.error('Could not copy text: ', err);
    });
}