var express = require('express');
var fs = require('fs');
var request = require('request');
var cheerio = require('cheerio');
var app     = express();
var readline = require('readline');

app.get('/scrape', function(req, res){
    var rl = readline.createInterface({
      input: fs.createReadStream('./crawler/data/sources.txt')
    });

    rl.on('line', function(line) {

      var url = line.replace(/^\s+|\s+$/g, '');
      request(url, function(error, response, html){

          if(!error){
            console.log(url);
              var $ = cheerio.load(html);
              var content = $('.lyrics').text();
              content = content.replace(/^\s+|\s+$/g, '');

              fs.appendFile('./crawler/data/results.txt', content, function(err) {
                if (err) {
                  console.log('ERROR');
                }
              });
          } else {
            console.log('error :/');
          }
      })
    });

    rl.on('close', function() {
      // res.send('FINISHED');
    })

});


app.listen('8081');

console.log('Magic happens on port 8081');

exports = module.exports = app;