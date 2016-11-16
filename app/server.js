import Express from "express";
import child_process from "child_process";
import bodyParser from "body-parser";
import cors from "cors";

const app = new Express();
app.use(cors());
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true, }));

const port = process.env.API_PORT || 3001;
const env = process.env.NODE_ENV || "prod";

const model_dir = "results/rnn/1476859465";

app.set("port", port);
app.post("/api/predict", (req, res) => {
  // Retrieve params
  const params = req.body;

  // Sanitize
  const inputs = params.inputs;
  const number = parseInt(params.number, 10) || 0;
  let cmdParams = {
    temperature: Math.max(parseFloat(params.temperature) || 0, 0.01),
    top_k: Math.max(parseInt(params.topk, 10) || 0, 1),
  }
  if (params.kind === 'para') {
    cmdParams['nb_para'] = number;
  } else if (params.kind === 'sentence') {
    cmdParams['nb_sentence'] = number;
  } else {
    cmdParams["nb_word"] = number;
  }

  // Build cmd
  let cmd = `cd .. && python predict-rnn.py --model_dir ${model_dir} --inputs "${inputs}"`;
  for (var key in cmdParams) {
    if (cmdParams.hasOwnProperty(key)) {
      cmd += ` --${key} ${cmdParams[key]}`;
    }
  }
  if (params.random === 'true' || params.random === true) {
    cmd += ' --random'
  }
  // console.log(cmd)

  // Python call
  const re = /__BBB_START__((.|\n|\r)*)__BBB_END__/;
  child_process.exec(cmd, (error, stdout) => {
    if (error) {
        console.error(`exec error: ${error}`);
        res.json({
          "output": "",
        });
        return;
    }

    const boobabotJson = re.exec(stdout)[1].trim();
    // console.log(boobabotJson);
    res.json(JSON.parse(boobabotJson));
  });

  return;
});

app.listen(app.get("port"), (err) =>{
  if (err) {
    return console.error(err);
  }
  console.info(`Server running on http://localhost:${app.get("port")} [${env}]`);
});
