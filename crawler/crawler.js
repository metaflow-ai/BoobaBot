var fs = require('fs');
var request = require('request');
var cheerio = require('cheerio');
var readline = require('readline');


raw_output_file = './crawler/data/raw_results.txt'
output_file = './crawler/data/results.txt'

// Cleaning existing files
if (fs.existsSync(raw_output_file)) {
  fs.unlink(raw_output_file)
}
if (fs.existsSync(output_file)) {
  fs.unlink(output_file)
}

var rl = readline.createInterface({
  input: fs.createReadStream('./crawler/data/sources.txt')
});

rl.on('line', function(url) {
  var url = url.trim();
  request(url, function(error, response, html){

    if(error){
      console.log('!!!Error in request for ' + url)
      return
    }

    console.log("HTML received for " + url);
    var $ = cheerio.load(html);
    var content = $('.lyrics').text();

    fs.appendFile(raw_output_file, content, function(err) {
      if (err) {
        console.log('ERROR outputing raw data:', err);
        return
      }
      content = content.replace(/^\s+|\s+$/gi, '');
      content = content.replace(/.*Genius.*|.*googletag.*/gi, '');
      content = content.replace(/\[.*\]/gi, '');

      fs.appendFile(output_file, content, function(err) {
        if (err) {
          console.log('ERROR outputing cleaned data:', err);
        }
      });
    });
  })
});

rl.on('close', function() {
  // res.send('FINISHED');
})