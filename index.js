var trainDataInputer = {
    trainDataInputerElement : document.getElementById("trainingDataInputer"),
    xCount : 0,
    yCount : 0,
    setCount : 0,

    /**
     * xCount, yCount, setCount 내용으로 html 요소들을 갱신합니다.
     */
    syncElements : function(){
        // 큰것부터 함
        {// setCount 갱신
            while(this.setCount > this.trainDataInputerElement.children.length){
                let newDataInputer_dataSetElement = document.createElement("section");
                newDataInputer_dataSetElement.setAttribute("class", "dataInputer_dataSet");
                newDataInputer_dataSetElement.setAttribute("data-index", this.trainDataInputerElement.children.length)

                let newDataInputer_dataSet_xElement = document.createElement("section");
                let newDataInputer_dataSet_yElement = document.createElement("section");
                newDataInputer_dataSet_xElement.setAttribute("class", "dataInputer_dataSet_x");
                newDataInputer_dataSet_yElement.setAttribute("class", "dataInputer_dataSet_y");
                newDataInputer_dataSetElement.appendChild(newDataInputer_dataSet_xElement);
                newDataInputer_dataSetElement.appendChild(newDataInputer_dataSet_yElement);
                this.trainDataInputerElement.appendChild(newDataInputer_dataSetElement);
            }
            while(this.setCount < this.trainDataInputerElement.children.length){
                this.trainDataInputerElement.removeChild(this.trainDataInputerElement.lastChild);
            }
        }
        {// input x 갱신
            for(let i = 0; i < this.setCount; i++){
                let setCount = this.trainDataInputerElement.children[i];
                let dataInputer_dataSet_xElement = setCount.children[0];
                while(this.xCount > dataInputer_dataSet_xElement.children.length){
                    let newDataInputer_dataSet_x_inputElement = document.createElement("input");
                    newDataInputer_dataSet_x_inputElement.setAttribute("type", "number");
                    newDataInputer_dataSet_x_inputElement.setAttribute("class", "dataInputer_dataSet_x_input");
                    newDataInputer_dataSet_x_inputElement.setAttribute("data-index", dataInputer_dataSet_xElement.children.length);
                    dataInputer_dataSet_xElement.appendChild(newDataInputer_dataSet_x_inputElement);

                }
                while(this.xCount < dataInputer_dataSet_xElement.children.length){
                    dataInputer_dataSet_xElement.removeChild(dataInputer_dataSet_xElement.lastChild);
                }
            }
        }
        {// y 갱신
            for(let i = 0; i < this.setCount; i++){
                let setCount = this.trainDataInputerElement.children[i];
                let dataInputer_dataSet_yElement = setCount.children[1];
                while(this.yCount > dataInputer_dataSet_yElement.children.length){
                    let newDataInputer_dataSet_y_inputElement = document.createElement("input");
                    newDataInputer_dataSet_y_inputElement.setAttribute("type", "number");
                    newDataInputer_dataSet_y_inputElement.setAttribute("class", "dataInputer_dataSet_y_input");
                    newDataInputer_dataSet_y_inputElement.setAttribute("data-index", dataInputer_dataSet_yElement.children.length);
                    dataInputer_dataSet_yElement.appendChild(newDataInputer_dataSet_y_inputElement);
                }
                while(this.yCount < dataInputer_dataSet_yElement.children.length){
                    dataInputer_dataSet_yElement.removeChild(dataInputer_dataSet_yElement.lastChild);
                }
            }
        }
        //스타일 갱신
        let tmp = this.xCount + this.yCount;
        tmp += tmp == 0;

        for(let i = 0; i < this.trainDataInputerElement.children.length; i++){
            let trainDataSet = this.trainDataInputerElement.children[i];
            let dataInputer_dataSet_xElement = trainDataSet.children[0];
            let dataInputer_dataSet_yElement = trainDataSet.children[1];
            dataInputer_dataSet_xElement.style.width = 100 * this.xCount / tmp + "%";
            dataInputer_dataSet_yElement.style.width = 100 * this.yCount / tmp + "%";
            for(let xIndex = 0; xIndex < dataInputer_dataSet_xElement.children.length; xIndex++){
                let dataInputer_dataSet_x_inputElement = dataInputer_dataSet_xElement.children[xIndex];
                dataInputer_dataSet_x_inputElement.style.width = 100 / this.xCount - 2 + "%";
            }
            for(let yIndex = 0; yIndex < dataInputer_dataSet_yElement.children.length; yIndex++){
                let dataInputer_dataSet_y_inputElement = dataInputer_dataSet_yElement.children[yIndex];
                dataInputer_dataSet_y_inputElement.style.width = 100 / this.yCount - 2 + "%";
            }
        }
    },
    appendX : function(){
        this.xCount ++;
        this.syncElements();
    },
    removeX : function(){
        this.xCount --;
        this.xCount *= (this.xCount > 0);
        this.syncElements();
    },
    appendY : function(){
        this.yCount ++;
        this.syncElements();
    },
    removeY : function(){
        this.yCount --;
        this.yCount *= (this.yCount > 0);
        this.syncElements();
    },
    appendData : function(){
        this.setCount++;
        this.syncElements();
    },
    removeData : function(){
        this.setCount--;
        this.setCount *= (this.setCount > 0);
        this.syncElements();
    }
};

var networkSetter = {
    networkSetterElement: document.getElementById("networkSetter"),
    layerCount: 0,

    syncElements: function(){
        while(this.layerCount > this.networkSetterElement.children.length){
            let newNetworkSetter_inputElement = document.createElement("input");
            newNetworkSetter_inputElement.setAttribute("type", "number");
            newNetworkSetter_inputElement.setAttribute("class", "networkSetter_input");
            this.networkSetterElement.appendChild(newNetworkSetter_inputElement);
        }
        while(this.layerCount < this.networkSetterElement.children.length){
            this.networkSetterElement.removeChild(this.networkSetterElement.lastChild);
        }

        
        for(let i = 0; i < this.layerCount; i++){
            let networkSetter_inputElement = this.networkSetterElement.children[i];
            networkSetter_inputElement.style.width = 100 / this.layerCount - 2 + "%";
        }
    },
    appendLayer: function(){
        this.layerCount ++;
        this.syncElements();
    },
    removeLayer: function(){
        this.layerCount --;
        this.layerCount = (this.layerCount > 0) * this.layerCount;
        this.syncElements();
    }
};

