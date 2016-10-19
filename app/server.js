import Express from 'express';
import child_process from 'child_process';
import bodyParser from 'body-parser';

const app = new Express();
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true }));

const port = process.env.API_PORT || 3001
const env = process.env.NODE_ENV || 'prod'

const model_dir = 'results/rnn/1476859465'

app.set('port', port)
app.post('/api/predict', (req, res) => {
  // Retrieve params
  const params = req.body;
  console.log(params)
  const inputs = "t'as";

  // Python call
  const re = /__BBB_START__((.|\n|\r)*)__BBB_END__/;
  const cmd = `cd .. && python predict-rnn.py --model_dir ${model_dir} --inputs "${inputs}"`
  child_process.exec(cmd, (error, stdout, stderr) => {
    if (error) {
        console.error(`exec error: ${error}`);
        res.json({
          'output': ''
        })
        return;
    }
    
    const boobabotJson = re.exec(stdout)[1].trim();
    res.json(JSON.parse(boobabotJson))
  })

  return;
})

app.listen(app.get('port'), (err) =>{
  if (err) {
    return console.error(err)
  }
  console.info(`Server running on http://localhost:${app.get('port')} [${env}]`)
})