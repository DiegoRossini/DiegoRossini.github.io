const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const cors = require('cors');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env.txt') });

const app = express();
const port = process.env.PORT || 3000;

// Serve static files
app.use(cors());
app.use(express.static(path.join(__dirname, '../assets/js')));
app.use(express.static(path.join(__dirname, '../assets/css')));
app.use('/images', express.static(path.join(__dirname, '../images')));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../../index.html'));
});

app.post('/send-email', (req, res) => {
    const { name, email, message } = req.body;

    const transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
            user: process.env.EMAIL_USER,
            pass: process.env.EMAIL_PASS
        }
    });

    const mailOptions = {
        from: email,
        to: process.env.EMAIL_USER,
        subject: `Message from ${name}`,
        text: message
    };

    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.error('Error sending email:', error);
            return res.status(500).send('Failed to send email');
        }
        console.log('Email sent:', info.response);
        res.send('Email sent successfully');
    });
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${port}/`);
});
