const express = require("express");
const app = express();
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const multer = require("multer");
const cors = require("cors");
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "./frontend_input");
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      path.parse(file.originalname).name +
        "-" +
        uniqueSuffix +
        path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

app.use(express.json());
app.use(cors());

app.post("/", upload.any(), (req, res) => {
  req.files = req.files[0];
  const filename = req.files.filename;
  const python = spawn("python", [`Calorie_Finding.py`, "--name", filename]);
  python.on("close", code => {
    const data = fs.readFileSync(`./outputs.txt`, {
      encoding: "utf8",
      flag: "r",
    });
    console.log("sent");
    res.send(data);
  });
});

app.get("/", (req, res) => {
  res.send("Home");
});
app.listen(4000, () => {
  console.log("Listening on port 4000");
});
