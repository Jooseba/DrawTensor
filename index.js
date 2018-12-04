import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';
import regeneratorRuntime from "regenerator-runtime";

import {MnistData} from './data';
import { deflateRaw } from 'zlib';
//import { drawToCanv } from '/draw.js';
//import { clearCanvas } from '/draw.js';
//import * from '/drawing.js'

var model;

//Luodaan tila tekstit
function createLogEntry(entry) {
    document.getElementById('log').innerHTML += '<span>' + entry + '</span>' + ' | ';
}

//luodaan modelit
function createModel() {
    createLogEntry('Create model ...');
    model = tf.sequential();
    createLogEntry('Model created');

    createLogEntry('Add layers ...');
    model.add(tf.layers.conv2d({  //layer 1
        inputShape: [28, 28, 1], //kuvan koko pixeleinä
        kernelSize: 5,  // kuinka suuri alue kerralla analysoidaan, tässä tapauksessa 5x5
        filters: 8,  //filtterien määrä, kuinka mota kuvaa, kuvasta
        strides: 1, //kuinka monen pixelin välein kerneli vertaus tehdään
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({ //Layer 2, luodaan maxpooling layer
        poolSize: [2,2],    //kuva tutkitaan pooleissa, ja otetaan niiden maximi arvot mitä niissä esiintyy
        strides: [2,2]
    }));

    model.add(tf.layers.conv2d({ //layer 3, samanlainen kuin 1 layer, mutta luodaan 16 filtteriä
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({ //Layer 4, samanlainen kuin 2 layeri
        poolSize: [2,2],
        strides: [2,2]
    }));

    model.add(tf.layers.flatten()); //5 layerissä tehään flattenim eli neuroverkko järjestetään riviin

    model.add(tf.layers.dense({ //Denselayer on viimeinen layeri, missä on luvut 0-9 johtopäätökset.
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }));

    createLogEntry('Layers created');

    createLogEntry('Start compiling ...'); //Määritetään loss function, optimizer ja metrics
    model.compile({
        optimizer: tf.train.sgd(0.15),
        loss: 'categoricalCrossentropy'
    });
    createLogEntry('Compiled');
}

let data;
async function load() {
    createLogEntry('Loading MNIST data ...'); //ladataan googlen minst vertailu kanta,
    data = new MnistData();
    console.log(data);
    await data.load();
    createLogEntry('Data loaded successfully');
}

const BATCH_SIZE = 64;  //kuinka monta näytettä laitetaan neuroverkkoon
const TRAIN_BATCHES = 150;  //kuinkamonta kertaa ne käydään läpi

async function train() {    //AI opetetaan
    createLogEntry('Start training ...');
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        const batch = tf.tidy(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]); //modelit reshapetaan oikeaan kokoon
            return batch;
        });

        await model.fit(
            batch.xs, batch.labels, {batchSize: BATCH_SIZE, epochs: 1}
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }
    createLogEntry('Training complete');
}

async function main() {
    createModel();
    await load();
    await train();
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Ramdomly Select Test Data And Predict";
}

//Tämä funktio vertailee arvoja ja tekee päättää miksikä piirroksen tunnistaa
async function predict(batch) {

    tf.tidy(() => {
        //const input_value = Array.from(batch.labels.argMax(1).dataSync());
        //console.log(input_value);

        const div = document.createElement('div');
        div.className = 'prediction-div';

        //const output2 = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

        //const output2 =batch.xs.reshape([-1, 28, 28, 1]); 
        //console.log(output2);
        const output =  model.predict(tf.image.resizeBilinear(tf.fromPixels(document.getElementById('canvasDraw'), 1), [28,28], false).reshape([-1,28,28,1])); //muokataan kuva oikeanlaiseen muotoon.
        console.log(output);
        const prediction_value = Array.from(output.argMax(1).dataSync()); //Muokataan kuvan arrayt prediction numeroksi, eli kone tunnistaa kuvan.
        //const input_value = Array.from(output);
        const drawenimg = tf.image.resizeBilinear(tf.fromPixels(document.getElementById('canvasDraw'), 1), [28,28], false).reshape([-1,28,28,1]);
       // const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);
        const image = drawenimg;
        console.log(image);
        console.log(output.argMax(1));
        

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const label = document.createElement('div');
       // label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;

      /*  if (prediction_value - input_value == 0) {
            label.innerHTML += '<br>Value recognized successfully';
        } else {
            label.innerHTML += '<br>Recognition failed!'
        }*/

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}

function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

document.getElementById('selectTestDataButton').addEventListener('click', async (el,ev) => {
    const batch = data.nextTestBatch(1); //haetaan dataa googlen kannasta
    await predict(batch);
    await clearCanvas_simple(); 
});

main();

var context = document.getElementById('canvasDraw').getContext("2d");

//var canvasWidth = 280;
//var canvasHeight = 280;

var canvasWidth = 140;
var canvasHeight = 140;

$('#canvasDraw').mousedown(function(e){
var mouseX = e.pageX - this.offsetLeft;
var mouseY = e.pageY - this.offsetTop;
paint = true;
addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
redraw();

});

$('#canvasDraw').mousemove(function(e){
if(paint){
  addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
  redraw();
}
});

$('#canvasDraw').mouseup(function(e){
paint = false;
});

$('#clearCanvasSimple').mousedown(function(e)
{
clearCanvas_simple(); 
});

$('#sendCanvasSimple').mousedown(function(e)
{
const canvaD = document.getElementById('canvasDraw');
var dImage = tf.fromPixels(canvaD, 1);
let reImg = tf.image.resizeBilinear(dImage, [28, 28], true);
let daImage = reImg.reshape([28,28,1]);
console.log(daImage);
});

function clearCanvas_simple()
{
    clickX = new Array();
    clickY= new Array();
    clickDrag = new Array();
context.clearRect(0, 0, canvasWidth, canvasHeight);
}

$('#canvasDraw').mouseleave(function(e){
paint = false;
});

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

function addClick(x, y, dragging)
{
clickX.push(x);
clickY.push(y);
clickDrag.push(dragging);
}

function redraw(){
context.strokeStyle = "#fff";
context.lineJoin = "round";
context.lineWidth = 15;
          
for(var i=0; i < clickX.length; i++) {		
  context.beginPath();
  if(clickDrag[i] && i){
    context.moveTo(clickX[i-1], clickY[i-1]);
   }else{
     context.moveTo(clickX[i]-1, clickY[i]);
   }
   context.lineTo(clickX[i], clickY[i]);
   context.closePath();
   context.stroke();
}
}