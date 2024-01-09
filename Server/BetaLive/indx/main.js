// ALIAS DEFINITION
const Live2DModel = PIXI.live2d.Live2DModel,
Application = PIXI.Application,
live2d = PIXI.live2d;
//Sprite = PIXI.Sprite.from,

class Waifu{
    'use strict'
    constructor(waifuPath,app){
        this.waifuPath = waifuPath;
        //this.windowSize = {width:1024,height:2048};
        this.app = app;//this.createApp();
        this.syncLoadModel();
    }
    appendToLogInfo(text){
        document.getElementById("log").innerHTML += `${text}\n`; 
    }
    post(){
        // este metodo se encargara de si se cargo el modelo aplicar transformaciones y informacion de logeo ademas
        // de desplegarlo
        this.appendToLogInfo(this.model);

        this.app.stage.addChild(this.model);
        this.setModel();
        this.draggable();
        console.log(this.model.hitAreas);
        //this.addHitAreas();
        this.onInteraction();

    }
    syncLoadModel(){
        /// cargar modelo de forma sincrona
        this.model = Live2DModel.fromSync(this.waifuPath,{ onError: console.warn });
        this.model.once('ready', ()=>this.post());
        // cargamos el modelo

        //Live2DModel.from(this.waifuPath).then((value)=>{this.model = value},(rejected)=>{this.model = undefined; this.appendToLogInfo(`Error loading the model \n :C sorry ... \n ${rejected}`)})
        //setTimeout(()=>{Live2DModel.from(this.waifuPath).then((value)=>{this.model = value;this.post();},(rejected)=>{this.appendToLogInfo(`Error loading the model \n :C sorry ... \n ${rejected}`)})},time);
    }
    setModel(){
        const scaleX = (innerWidth * 0.4) / this.model.width;
        const scaleY = (innerHeight * 0.8) / this.model.height;
    
        // fit the window
        this.model.scale.set(Math.min(scaleX, scaleY));    
        this.model.rotation = Math.PI;
        this.model.skew.x = Math.PI;
        //this.model.x = 2048;
        //this.model.y = 2048;
        //this.model.scale.set(2, 2);
        this.model.anchor.set(0.5, 0.5);
    }
    draggable() {
        this.model.buttonMode = true;
        this.model.on("pointerdown", (e) => {
          this.model.dragging = true;
          this.model._pointerX = e.data.global.x - this.model.x;
          this.model._pointerY = e.data.global.y - this.model.y;
        });
        this.model.on("pointermove", (e) => {
          if (this.model.dragging) {
            this.model.position.x = e.data.global.x - this.model._pointerX;
            this.model.position.y = e.data.global.y - this.model._pointerY;
          }
        });
        this.model.on("pointerupoutside", () => (this.model.dragging = false));
        this.model.on("pointerup", () => (this.model.dragging = false));
      }
    onClickArea(hitAreaName='body',executeMotion='TapBody'){
        this.appendToLogInfo(this.model.hitAreas);

        this.model.on('hit', (hitAreas) => {
            if (hitAreas.includes(hitAreaName)) {
                //console.log(`hit ${hitAreas} ${executeMotion}`)
                this.motion(executeMotion);
                //this.model.expression(0);
              
            }
          });
    }
    onInteraction(){
        //this.model.expression('goth_5');
        //this.onClickArea('Breasts','Breasts#1');
        //this.onClickArea('Head','Head');
        //this.addHitAreas();
    }
}
function createApp(canvasId){
    return new Application({
        view: document.getElementById(canvasId),
        autoStart: true,
        //width:2048,height:2048,
        resizeTo: window,
        backgroundColor: 0x333333
      });// create the pixi app
}

function main(){
    const app = createApp('canvas');
    let instance = new Waifu('./goth/goth.model.json',app);
    let ui_instance = new UI_GAME(app);
    window.PIXI = PIXI;
    //console.log(typeof instance.model.motion)
}