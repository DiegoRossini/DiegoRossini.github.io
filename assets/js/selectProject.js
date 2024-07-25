document.addEventListener('DOMContentLoaded', function() {
    // Get references to the dropdown and button elements
    var selectElement = document.getElementById('demo-category');
    var goButton = document.getElementById('go-button');
    
    // Add click event listener to the button
    goButton.addEventListener('click', function() {
        // Get the selected option's value
        var selectedValue = selectElement.value;
        
        // Check if a valid option is selected
        if (selectedValue) {
            // Redirect to the URL specified in the selected option
            window.location.href = selectedValue;
        } else {
            // Alert the user if no valid option is selected
            alert('Please select a project from the dropdown.');
        }
    });
});
