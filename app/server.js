const express = require('express');
const cors = require('cors');
const fs = require('fs');
const app = express();

app.use(cors());

app.get('/api/status', (req, res) => {
    const data = {
        sound: "s",
        direction: -1
    };
    res.json(data);
});

app.listen(3000, () => console.log('Pi data streaming on port 3000'));

function exportJson(sound, direction) {
    const output = JSON.stringify({ sound, direction });
    fs.writeFileSync('output.json', output);
}