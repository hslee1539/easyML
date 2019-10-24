// 작은것에서 큰 순으로 진행.

function gaussianRandom(){
    var v1, v2, s;
    do{
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    }while(s >= 1 || s == 0);
    return v1 * Math.sqrt( -2 * Math.log(s) / s);
}

/**
 * 레이어의 입출력 데이터를 가지고 있는 클래스입니다.
 */
class Layer{
    constructor(){
        /**
        * 레이어의 입력을 미분한 값이 옵니다.
        */
        this.dx = [0];
        /**
        * 레이어의 결과값입니다.
        */
        this.out = [0];
        /**
         * 데이터의 수 입니다.
         */
        this.dataN = 0;
        this.inputN = 0;
        this.outputN = 0;
        
    }
    syncDataN(dataN){
        if(this.dataN != dataN){
            this.dataN = dataN;
            this.dx = new Array(this.dataN * this.inputN);
            this.out = new Array(this.dataN * this.outputN);
        }
    }
    syncInputN(inputN){
        if(this.inputN != inputN){
            this.inputN = inputN;
            this.dx = new Array(this.dataN * this.inputN);
        }
    }
    syncOutputN(outputN){
        if(this.outputN != outputN){
            this.outputN = outputN;
            this.out = new Array(this.dataN * this.outputN)
        }
    }
    /**
     * 
     * @param {Number[]} out
     */
    copyOut(out){
        for (let index = 0; index < this.out.length; index++){
            this.out[index] = out[index];
        }
    }
}

class InputLayer{
    constructor(){
        this.layer = new Layer();
    }
    syncDataN(dataN){
        this.layer.syncDataN(dataN);
    }
    syncOutputN(outputN){
        this.layer.syncOutputN(outputN);
    }
    copyX(x){
        this.layer.copyOut(x);
    }
}

/**
 * Fully connected layer의 가중치 데이터를 관리하는 클래스입니다.
 */
class FCWeight{
    /**
     * Fully connected layer의 가중치 데이터 객체를 생성합니다.
     */
    constructor(){
        this.w = [0];
        this.b = [0];
        this.momntW = [0];
        this.momntB = [0];
        this.size = 0;
        this.inputN = 0;
        this.outputN = 0;
    }
    syncInputN(inputN){
        if(this.inputN != inputN){
            this.inputN = inputN;
            this.size = this.inputN * this.outputN
            this.w = new Array(this.size);
            this.momntW = new Array
            for (let index = 0; index < this.size; index++) {
                this.w[index] = gaussianRandom() / Math.sqrt(this.inputN / 2);
                this.momntW[index] = 0;
            }
        }
    }
    syncOutputN(outputN){
        if(this.outputN != outputN){
            this.outputN = outputN;
            this.size = this.inputN * this.outputN;
            this.w = new Array(this.size);
            this.b = new Array(this.outputN);
            this.momntW = new Array(this.size);
            this.momntB = new Array(this.outputN)
            for (let index = 0; index < this.size; index++) {
                //he 초기화
                this.w[index] = gaussianRandom() / Math.sqrt(this.inputN / 2);
                this.momntW[index] = 0;
            }
            for (let index = 0; index < this.outputN; index ++){
                this.b[index] = 0;
                this.momntB[index] = 0;
            }
        }
    }
}


class FCRLayer{
    /**
     * Relu(Batchnorm(Fully connected)) layer 객체를 생성합니다.
     */
    constructor(){
        this.weight = new FCWeight();
        this.layer = new Layer();
        this.activationDx = [0];
        this.dispersion = [0];
    }
    syncDataN(dataN){
        if(this.layer.dataN != dataN){
            this.layer.syncDataN(dataN);
            this.batchnormOut = new Array(this.layer.dataN * this.layer.outputN);
            this.batchnormDx = new Array(this.layer.dataN * this.layer.outputN);
        }
    }
    syncInputN(inputN){
        this.layer.syncInputN(inputN);
        this.weight.syncInputN(inputN);
    }
    syncOutputN(outputN){
        if(this.layer.outputN != outputN){
            this.layer.syncOutputN(outputN);
            this.weight.syncOutputN(outputN);
            this.batchnormOut = new Array(this.layer.dataN * this.layer.outputN);
            this.batchnormDx = new Array(this.layer.dataN * this.layer.outputN);
            this.dispersion = new Array(this.layer.outputN);
        }
    }
    /**
     * 순전파 계산합니다.
     * @param {Layer} forwardLayer 
     * 앞 레이어의 Layer 객체를 받습니다.
     */
    forward(forwardLayer){
        this.forwardLayer = forwardLayer;
        for(let outputIndex = 0; outputIndex < this.layer.outputN; outputIndex++){
            let mean = 0;
            for(let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex ++){
                let tmp = 0;
                for(let productIndex = 0; productIndex < this.layer.inputN; productIndex++){
                    tmp += forwardLayer.out[dataIndex * this.layer.inputN + productIndex] * this.weight.w[outputIndex + this.layer.outputN * productIndex];
                }
                tmp += this.weight.b[outputIndex];
                // ReLU layer
                tmp *= tmp > 0;
                this.layer.out[dataIndex * this.layer.outputN + outputIndex] = tmp;
                
                mean += tmp;
            }
            // 평균
            mean /= this.layer.dataN;

            // 분산
            this.dispersion[dataIndex] = 0;
            for(let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex ++){
                this.dispersion[outputIndex] += (this.layer.out[dataIndex * this.layer.outputN + outputIndex] - mean) * (this.layer.out[dataIndex * this.layer.outputN + outputIndex] - mean);
            }

            // bachNorm
            this.dispersion[outputIndex] = Math.sqrt(this.dispersion[outputIndex] / N + 0.00000001);
            for(let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex ++){
                this.layer.out[dataIndex * this.layer.outputN + outputIndex] = (this.layer.out[dataIndex * this.layer.outputN + outputIndex] - mean) / this.dispersion[outputIndex];
            }
        }

        /*
        for (let out_index = 0; out_index < this.layer.out.length; out_index++) {
            // fully connected
            var pass_inputOut_index = Math.floor(out_index / this.weight.outputN) * this.weight.inputN; // 정수 나누기가 있으면 Math.floor 제거 가능
            var pass_w_index = out_index % this.weight.outputN;
            var tmp = 0;
            
            for (let productIndex = 0; productIndex < this.weight.inputN; productIndex++) {
                tmp += forwardLayer.out[pass_inputOut_index + productIndex] * this.weight.w[pass_w_index + this.weight.outputN * productIndex];
            }
            tmp += this.weight.b[pass_w_index];
            // relu
            this.layer.out[out_index] = tmp * (tmp > 0);
        }*/
        return true
    }
    /**
     * 역전파 계산합니다.
     * @param {Layer} backwardLayer 
     * 앞 레이어의 Layer 객체를 받습니다,
     */
    backward(backwardLayer){
        // 기존 out 손실
        // relu dx로 사용 됨.
        //for (let index = 0; index < this.layer.out.length; index++) {
        //    this.layer.out[index] = (this.layer.out[index] > 0) * backwardLayer.dx[index];
        //}
        // out은 그래픽화 하기 위해 activation_dx로 
        this.backwardLayer = backwardLayer;
        let dataN = this.layer.dataN;
        let outputN = this.layer.outputN;

        for (let outputIndex = 0; outputIndex < outputN; outputIndex++){
            let tmp1 = 0;
            for(let dataIndex = 0; dataIndex < dataN; dataIndex++){
                tmp1 += backwardLayer.dx[dataIndex * outputN + outputIndex] * this.layer.out[dataIndex * outputN + outputIndex];
            }
            let tmp2 = 0;
            for(let dataIndex = 0; dataIndex < dataN; dataIndex++){
                tmp2 += tmp1 * this.layer.out[dataIndex * outputN + outputIndex] / dataN - backwardLayer.dx[dataIndex * outputN + outputIndex];
            }
            tmp2 /= dataN;
            
            for(let dataIndex = 0; dataIndex < dataN; dataIndex++){
                this.activationDx[dataIndex * outputN + outputIndex] = (tmp2 - tmp1 * this.layer.out[dataIndex * outputN + outputIndex] / dataN + backwardLayer.dx[dataIndex * outputN + outputIndex]) / this.dispersion[outputN];
            }
        }

        for (let dx_index = 0; dx_index < this.layer.dx.length; dx_index++) {
            var pass_dout_index = Math.floor(dx_index / this.weight.inputN) * this.weight.outputN;
            var pass_w_index = dx_index % this.weight.inputN * this.weight.outputN;
            var tmp = 0;
            for (let productIndex = 0; productIndex < this.weight.outputN; productIndex++) {
                // 기존 out이 아님
                this.activationDx[pass_dout_index + productIndex] = (this.layer.out[pass_dout_index + productIndex] > 0) * backwardLayer.dx[pass_dout_index + productIndex];
                tmp += this.activationDx[pass_dout_index + productIndex] * this.weight.w[pass_w_index + productIndex];
            }
            this.layer.dx[dx_index] = tmp;
        }
        return true
    }

    update(learning_rate = 0.02){
        // w update
        for (let index = 0; index < this.weight.w.length; index++) {
            var dw1 = index % this.weight.outputN;
            var dw0 = Math.floor(index / this.weight.outputN);
            var dw = 0;
            for (let x0 = 0; x0 < this.forwardLayer.dataN; x0++){
                dw += this.forwardLayer.out[x0 * this.forwardLayer.outputN + dw0] * this.activationDx[x0 * this.weight.outputN + dw1];
            }
            //ada
            this.weight.momntW[index] += dw * dw;
            this.weight.w[index] -= learning_rate * dw / (Math.sqrt(this.weight.momntW[index]) + 0.00000001);
        }
        // b update
        for (let db_index = 0; db_index < this.weight.outputN; db_index ++){
            var db = 0;
            for (let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex++ ){
                db += this.activationDx[dataIndex * this.weight.outputN + db_index];
            }
            //ada
            this.weight.momntB[db_index] += db * db;
            this.weight.b[db_index] -= learning_rate * db / (Math.sqrt(this.weight.momntB[db_index]) + 0.00000001);
        }
        return true;
    }
}

class OutputLayer{
    constructor(){
        this.layer = new Layer();
        this.weight = new FCWeight();
        this.loss = 0;
        this.activationDx = [0];
    }
    syncDataN(dataN){
        if(this.layer.dataN != dataN){
            this.layer.syncDataN(dataN);
            this.activationDx = new Array(this.layer.dataN * this.layer.outputN);
        }
    }
    syncInputN(inputN){
        this.layer.syncInputN(inputN);
        this.weight.syncInputN(inputN);
    }
    syncOutputN(outputN){
        if(this.layer.outputN != outputN){
            this.layer.syncOutputN(outputN);
            this.weight.syncOutputN(outputN);
            this.activationDx = new Array(this.layer.dataN * this.layer.outputN);
        }
    }
    /**
     * 
     * @param {Layer} forwardLayer 
     */
    forward(forwardLayer){
        this.forwardLayer = forwardLayer;
        // fully connected
        for (let out_index = 0; out_index < this.layer.out.length; out_index++) {
            var pass_inputOut_index = Math.floor(out_index / this.weight.outputN) * this.weight.inputN; // 정수 나누기가 있으면 Math.floor 제거 가능
            var pass_w_index = out_index % this.weight.outputN;
            var tmp = 0;
            
            for (let productIndex = 0; productIndex < this.weight.inputN; productIndex++) {
                tmp += forwardLayer.out[pass_inputOut_index + productIndex] * this.weight.w[pass_w_index + this.weight.outputN * productIndex];
            }
            // f.c. 결과를 임시 저장
            this.layer.out[out_index] = tmp + this.weight.b[pass_w_index];
        }
        // softmax
        for (let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex++){
            var passIndex = dataIndex * this.layer.outputN;
            var sigma = 0;
            for (let c = 0; c < this.layer.outputN; c++){
                this.layer.out[passIndex + c] = Math.exp(forwardLayer.out[passIndex + c]);
                sigma += this.layer.out[passIndex + c];
            }
            for (let c = 0; c < this.layer.outputN; c++){
                this.layer.out[passIndex + c] /= sigma;
            }
        }
        
        return true
    }
    /**
     * 역전파 계산합니다.
     * @param {Number[]} y 
     * 정답을 받습니다.
     */
    backward(y){
        // 기존 out 손실
        // relu dx로 사용 됨.
        //for (let index = 0; index < this.layer.out.length; index++) {
        //    this.layer.out[index] = (this.layer.out[index] > 0) * backwardLayer.dx[index];
        //}
        // out은 그래픽화 하기 위해 activation_dx로 
        this.y = y;
        for (let dx_index = 0; dx_index < this.layer.dx.length; dx_index++) {
            var pass_dout_index = Math.floor(dx_index / this.weight.inputN) * this.weight.outputN;
            var pass_w_index = dx_index % this.weight.inputN * this.weight.outputN;
            var tmp = 0;
            for (let productIndex = 0; productIndex < this.weight.outputN; productIndex++) {
                this.activationDx[pass_dout_index + productIndex] = this.layer.out[pass_dout_index + productIndex] - y[pass_dout_index + productIndex];
                tmp += this.activationDx[pass_dout_index + productIndex] * this.weight.w[pass_w_index + productIndex];
            }
            this.layer.dx[dx_index] = tmp;
        }
        return true
    }

    update(learning_rate = 0.02){
        // w update
        for (let index = 0; index < this.weight.w.length; index++) {
            var dw1 = index % this.weight.outputN;
            var dw0 = Math.floor(index / this.weight.outputN);
            var dw = 0;
            for (let x0 = 0; x0 < this.forwardLayer.dataN; x0++){
                dw += this.forwardLayer.out[x0 * this.forwardLayer.outputN + dw0] * this.activationDx[x0 * this.weight.outputN + dw1];
            }
            //ada
            this.weight.momntW[index] += dw * dw;
            this.weight.w[index] -= learning_rate * dw / (Math.sqrt(this.weight.momntW[index]) + 0.00000001);
        }
        // b update
        for (let db_index = 0; db_index < this.weight.outputN; db_index ++){
            var db = 0;
            for (let dataIndex = 0; dataIndex < this.layer.dataN; dataIndex++ ){
                db += this.activationDx[dataIndex * this.weight.outputN + db_index];
            }
            //ada
            this.weight.momntB[db_index] += db * db;
            this.weight.b[db_index] -= learning_rate * db / (Math.sqrt(this.weight.momntB[db_index]) + 0.00000001);
        }
        return true;
    }
}

var network = {
    inputLayer : new InputLayer(),
    outputLayer : new OutputLayer(),
    hiddenLayers : [new FCRLayer()],
    xData : [0],
    yData : [0],
    xCount : 0,
    yCount : 0,
    setCount : 0,

    init : function(){
        this.hiddenLayer = [];
        this.xData = [];
        this.yData = [];
    },

    /**
     * 
     * @param {HTMLElement} inputElement 
     */
    syncXData: function(inputElement){
        let value = Number.parseFloat(inputElement.nodeValue);
        let dataSetElement = inputElement.parentElement.parentElement;
        let dataSet_xElement = dataSetElement.children[0];
        let setIndex = Number.parseInt(dataSetElement.getAttribute("data-index"));
        let xIndex = Number.parseInt(inputElement.getAttribute("data-index"));
        let xCount = dataSet_xElement.children.length;
        this.xData[xIndex + setIndex * xCount] = value;
    },

    sync : function(){
        this.inputLayer.syncDataN(this.setCount);
        this.outputLayer.syncDataN(this.setCount);
        this.inputLayer.syncOutputN(this.xCount);
        this.outputLayer.syncOutputN(this.yCount);
        let inputN = this.xCount;
        for (let i = 0; i < this.hiddenLayers.length; i++){
            let layer = this.hiddenLayers[i];
            layer.syncDataN(this.setCount);
            layer.syncInputN(inputN);
            inputN = layer.layer.outputN;
        }
        this.outputLayer.syncInputN(inputN);
        this.inputLayer.copyX(this.xData);
    },

    syncHidden : function(networkSetterID){
        let networkSetterElement = document.getElementById(networkSetterID);
        while(this.hiddenLayers.length < networkSetterElement.childElementCount){
            this.hiddenLayers.push(new FCRLayer());
        }
        while(this.hiddenLayers.length > networkSetterElement.childElementCount){
            this.hiddenLayers.pop();
        }
        for(let i = 0; i < this.hiddenLayers.length; i ++){
            let layer = this.hiddenLayers[i];
            let inputElement = networkSetterElement.children[i];
            layer.syncOutputN(Number.parseInt(inputElement.value));
        }
        this.sync();
    },

    syncXY : function(dataInputerID){
        let dataInputerElement = document.getElementById(dataInputerID);
        let setCount = dataInputerElement.childElementCount;
        let xCount_count = 0;
        let yCount_count = 0;
        for(let setIndex = 0; setIndex < setCount; setIndex++){
            let dataInputer_dataSetElement = dataInputerElement.children[setIndex];
            let dataInputer_dataSet_xElement = dataInputer_dataSetElement.children[0];
            let dataInputer_dataSet_yElement = dataInputer_dataSetElement.children[1];
            var xCount = dataInputer_dataSet_xElement.childElementCount;
            xCount_count += xCount;
            for(let xIndex = 0; xIndex < xCount; xIndex++){
                let dataInputer_dataSet_x_inputElement = dataInputer_dataSet_xElement.children[xIndex];
                this.xData[xIndex + setIndex * xCount] = Number.parseFloat(dataInputer_dataSet_x_inputElement.value);
            }
            var yCount = dataInputer_dataSet_yElement.childElementCount;
            yCount_count += yCount;
            for(let yIndex = 0; yIndex < yCount; yIndex++){
                let dataInputer_dataSet_y_inputElement = dataInputer_dataSet_yElement.children[yIndex];
                this.yData[yIndex + setIndex * yCount]= Number.parseFloat(dataInputer_dataSet_y_inputElement.value);
            }
        }
        if(setCount > 0){
            if(xCount != xCount_count / setCount){
                console.error("x가 일정하지 않음");
            }
            if(yCount != yCount_count / setCount){
                console.error('y가 일정하지 않음');
            }
            this.xCount = xCount;
            this.yCount = yCount;
            this.setCount = setCount;
        }
        this.sync();
    },

    train : function(epoch, learning_rate = 0.05){
        for(let e = 0; e < epoch; e++){
            let forwardLayer = this.inputLayer.layer;
            for(let l = 0; l < this.hiddenLayers.length; l++){
                let layer = this.hiddenLayers[l];
                layer.forward(forwardLayer);
                forwardLayer = layer.layer;
            }
            this.outputLayer.forward(forwardLayer);
            this.outputLayer.backward(this.yData);
            let backwardLayer = this.outputLayer.layer;
            for(let l = this.hiddenLayers.length - 1; l > -1; l--){
                let layer = this.hiddenLayers[l];
                layer.backward(backwardLayer);
                backwardLayer = layer.layer;

            }
            for(let l = 0; l < this.hiddenLayers.length; l++){
                let layer = this.hiddenLayers[l];
                layer.update(learning_rate);
            }
            this.outputLayer.update(learning_rate);
        }
        console.log(this.outputLayer.layer.out);
    }

}
// TODO: outputLayer.syncInputN()를 히든 레이어 설정때 해주자!
network.init();

/*
x = [0,0,0,1,1,0,1,1]
y = [0,1,1,0,1,0,0,1]



var l0 = new InputLayer();
l0.syncDataN(4);
l0.syncOutputN(2);
l0.copyX(x);
var l1 = new FCRLayer();
l1.syncDataN(4);
l1.syncInputN(2);
l1.syncOutputN(3);
var l2 = new OutputLayer();
l2.syncDataN(4);
l2.syncInputN(3);
l2.syncOutputN(2);

for(let i = 0; i < 10000; i ++){
    l1.forward(l0.layer);
    l2.forward(l1.layer);
    l2.backward(y);
    l1.backward(l2.layer);
    l1.update(0.05);
    l2.update(0.05);
    if(i % 2000 == 0){
        console.log(l2.layer.out);
    }
}
*/