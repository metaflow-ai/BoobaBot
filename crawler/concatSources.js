var concatenate = require('concatenate');

var artists = [
        "booba",
        "rohff",
        "kery-james",
        "soprano",
        "kaaris",
        "la-fouine",
        "youssoupha",
        "jul",
        "akhenaton",
        "sinik",
        "medine",
        "nekfeu",
        "mc-solaar",
        "alonzo",
        "lacrim",
        "disiz",
        "sefyu",
        "lefa",
        "joeystarr",
        "diams",
        "doc-gyneco",
        "orel-san",
        "sniper",
        "passi",
        "guizmo",
        "abd-al-malik",
        "oxmo-puccino",
        "lucio-bukowski",
        "dadoo",
        "gringe",
        "pnl",
        "salif",
        "vald",
        "lunatic",
        "seth-geko",
        "mafia-k1-fry",
        "negmarrons",
        "lalgerino"
];

var outputFile = './crawler/data/sources_all.txt'

var files = [];
for (var i = 0; i < artists.length; i++) {
  files.push("./crawler/data/sources/sources_" + artists[i] + ".txt");
}

concatenate.sync(files, outputFile);