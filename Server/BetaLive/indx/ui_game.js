
// require the pixi/ui

const loader = PIXI.Loader.shared,
resources = PIXI.Loader.shared.resources,
Sprite = PIXI.Sprite,
TextureCache = PIXI.utils.TextureCache,
Container = PIXI.Container;




class UI_GAME{
    constructor(app){
        this.currentLevel = 'level_main';
        this.app = app;
        this.jsonLevel = null;
        this.ui_file = './assets/ui/ui.json';
        this.a = 'analog';
        this.debug = true;
        this._spriteDict = {};
        // "./assets/UI_Silent/ui_file.json"
        //this.objectInteraction();
        this.loader = loader.add(this.ui_file).load(async ()=>this.seg(this));
        //this.setup();
    }
    
    async seg(){
        await this.loadUiSettings();//.then(response=>response).catch(rejected=>rejected);s
        this.assets = resources[this.ui_file].textures;
        await this.mapButtons();
        await this.mapBar();
        await this.mapSignalAnimation();
        //this._spriteDict["love_level"]["methadata"]["alter_state"]();

        /*
        let sp = new Sprite(this.assets['Buttons/Rect-Medium/PlayIcon/Default.png']);
        sp.eventMode = 'dynamic';
        sp.interactive = true;
        sp.cursor = 'pointer';
        console.log(sp.isInteractive)
        sp.on("pointerdown",(event)=>{this.texture = this.assets['Buttons/Rect-Medium/PlayIcon/Hover.png'];alert('clicked')});*/
        //this.app.stage.addChild(sp);


    }
    get jlevel(){
        return this.jsonLevel;
    }
    get getSpriteDict(){
        return this._spriteDict;
    }
    set updateSpritesDict(arg){
        /*
        arg -> {key:value}

         */
        this._spriteDict = {...this._spriteDict,...arg}
    }
    async loadUiSettings(){
        const response = await fetch(`./resources/${this.currentLevel}.json`);
        if(response.ok){
            this.jsonLevel = await response.json();
        }else{
            throw new Error("NET ERROR!");
        }
    }
    /*  EJEMPLOS DE METODOS DEFINIDOS POR EL USUARIO PARA CREAR BOTONES
    ArrowRightOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        setTimeout(async (ths)=>{sprite.texture = ths.assets[ButtonSettings.assetKey]},100,ths);
        // generar animacion
    }
    sound_OffOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        setTimeout(async (ths)=>{sprite.texture = ths.assets[ButtonSettings.assetKey]},100,ths);
    }
    sound_OnOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        setTimeout(async (ths)=>{sprite.texture = ths.assets[ButtonSettings.assetKey]},100,ths);
    }
    //alu
    */
   /*
    love_levelUpgradeProgress(ths,spriteBack,LoadSprite){
        //LoadSprite.width = LoadSprite.width;
        LoadSprite.width = LoadSprite.width *0.5;
        //setInterval(async (ths)=>{LoadSprite.width = LoadSprite.width * ths._spriteDict["love_level"]["methadata"]["status"];ths._spriteDict["love_level"]["methadata"]["status"] = 0.5},100,ths);
    }*/
    orgasmHook(ths,sprite){
        ths.app.stage.addChild(sprite);

    }
    async objectInteraction(){
        /*funcion para crear un objeto que mapeara la referencia de la una funcion callback con una accion es decir
        {"ArrowRight":ArrowRightOnClick}

        */
       /* EJEMPLO DE CALLBACKS PARA PROCESAR EVENTOS
       this.callBacks = {
        "ArrowRightOnClick":{"event":"click","hook":this.ArrowRightOnClick},
        "sound_OffOnClick":{"event":"click","hook":this.sound_OffOnClick},
        "sound_OnOnClick":{"event":"click","hook":this.sound_OnOnClick},
        "love_levelUpgradeProgress":{"hook":this.love_levelUpgradeProgress}
        "orgasmHook":{"hook":this.orgasmHook}
        "orgasmHook":{"hook":this.orgasmHook}
       }*/
       this.callBacks = {
       };
    }
    async mapButtons(){

        //console.log({...this.jsonLevel.buttons});
        await this.objectInteraction();
        console.log(this.callBacks);
        for(const [key,value] of Object.entries(this.jsonLevel.buttons)){
            let spriteTemporal = new Sprite(this.assets[value.assetKey]);
            //if(!(this.jsonLevel.SpecialButtons.includes(key))){
                    spriteTemporal.eventMode  = 'static';// para objetos que no se mueven
                    spriteTemporal.interactive = true
                    spriteTemporal.cursor = 'pointer';
                    spriteTemporal.anchor.set(0.5);
                    this._spriteDict[key] = {'sprite':spriteTemporal,'methadata':{'status':false},"type":"button"};
                    Object.keys(this.callBacks).forEach((callback)=>
                    {
                        if(callback.startsWith(key) === true){
                            spriteTemporal.on(this.callBacks[callback]["event"],(event)=>{this.callBacks[callback]["hook"](event,this,value,key,spriteTemporal)});
                            console.log(`${key} and ${this.callBacks[callback]["event"]}: mapped with ${this.callBacks[callback]["hook"]} \n ${callback}`);
                        }
                        // callback.startsWith(key) === true? spriteTemporal.on(this.callBacks[callback]["event"],(event)=>{this.callBacks[callback]["hook"](event)}):false;
                        
                    })
                    console.log(spriteTemporal)
                    spriteTemporal.x = value.x;
                    spriteTemporal.y = value.y;
                    //console.log(spriteTemporal.x)
                    this.app.stage.addChild(spriteTemporal);
                    //nval['SpriteHover'] = new Sprite(this.assets[value.assetHover]);
                    

            
        }

        // enlazamos el evento

    }
    async mapSignalAnimation(){
        /* signals like love or date or effects like in minecraft  */
        await this.objectInteraction();
        for(const [key,value] of Object.entries(this.jsonLevel.signals)){
            //console.log(value.animationsFrames);
            //console.log(this.loader)
            //console.log(key)
            //console.log(this.loader.resources['./assets/ui/ui.json']["data"]["animations"][key]);
            //console.log(typeof resources[this.ui_file].data.animations.orgasm[0])
            console.log(resources[this.ui_file].data.animations[key].forEach(txt=>txt))
            const signalSprite = new PIXI.AnimatedSprite(resources[this.ui_file].data.animations[key].map((texture)=>this.assets[texture]))//this.loader.resources['./assets/ui/ui.json']["data"]["animations"][key]);
            signalSprite.animationSpeed   = value.speed;
            // the status in this case is for hook the signal (make it visible)
            // inside the method you can make visible the signal or remove for a better preformance
            // at start the signalas will not be visibles or added to app.stage.addChild\
            Object.keys(this.callBacks).forEach((callback)=>
            {
                if(callback.startsWith(key) === true){
                    this._spriteDict[key] =  {'sprite':signalSprite,'methadata':{'status':false,"alter_state":()=>this.callBacks[callback]["hook"](this,signalSprite)},"type":"signal"};
                    console.log(`${key} and alter_state: mapped with ${this.callBacks[callback]["hook"]} \n ${callback}`);
                }
                // callback.startsWith(key) === true? spriteTemporal.on(this.callBacks[callback]["event"],(event)=>{this.callBacks[callback]["hook"](event)}):false;
                
            });
        }
    }
    async mapBar(){
        await this.objectInteraction();
        for(const [key,value] of Object.entries(this.jsonLevel.ProgressBar)){
            const spriteBack = new Sprite(this.assets[value.background]);
            const LoadSprite = new Sprite(this.assets[value.lineLoad]);
            const ContainerBack = new Container();
            const ContainerLoad  = new Container();
            ContainerBack.addChild(spriteBack);
            LoadSprite.height = value.LineLoadScale.h;
            LoadSprite.width = value.LineLoadScale.w * value.state;
            
            ContainerLoad.addChild(LoadSprite);
            ContainerLoad.x = value.LineLoadScale.line_height.x;
            ContainerLoad.y = value.LineLoadScale.line_height.y;
            ContainerBack.addChild(ContainerLoad);

            ContainerBack.x = value.x;
            ContainerBack.y = value.y;
            Object.keys(this.callBacks).forEach((callback)=>
            {
                if(callback.startsWith(key) === true){
                    this._spriteDict[key] = {'spriteBack':spriteBack,"spriteFont":LoadSprite,'methadata':{'status':value.status,"alter_state":()=>this.callBacks[callback]["hook"](this,spriteBack,LoadSprite)},"type":"loadBar"};
                    console.log(`${key} and alter_state: mapped with ${this.callBacks[callback]["hook"]} \n ${callback}`);
                }
                // callback.startsWith(key) === true? spriteTemporal.on(this.callBacks[callback]["event"],(event)=>{this.callBacks[callback]["hook"](event)}):false;
                
            });
            this.app.stage.addChild(ContainerBack);


        }

    }
    /*
    createObjectsDict(){
        // este metodo creara un diccionario de objetos sera usado para manejar de una manera mas amena los objetos que involucran a cada interfaz
    }*/
    setup(ths){
        //console.log(TextureCache)
        ths.assets = resources[ths.ui_file].textures;
        ths.button = ths.assets['Buttons/Rect-Medium/PlayIcon/Hover.png'];
        ths.spriteButton = new Sprite(ths.button);
        ths.spriteButton.x = 10;
        ths.spriteButton.y = 10;
        ths.app.stage.addChild(ths.spriteButton);

        //console.log(ths.button);
        

        //let button = new Sprite();

        //console.log(this.assets);
        //this.assets = resources[this.ui_file];//.textures;

    }
}
class MAIN_LEVEL extends UI_GAME{
    constructor(app){
        
        super(app)
        this.waitAnimation = (sprite,ths,ButtonSettings)=>setTimeout(async (ths)=>{sprite.texture = ths.assets[ButtonSettings.assetKey]},100,ths);
    }
    async objectInteraction(){
        /*funcion para crear un objeto que mapeara la referencia de la una funcion callback con una accion es decir
        
        */
       await super.objectInteraction();
       this.callBacks = {
        "ArrowRightOnClick":{"event":"click","hook":this.ArrowRightOnClick},
        "sound_OffOnClick":{"event":"click","hook":this.sound_OffOnClick},
        "sound_OnOnClick":{"event":"click","hook":this.sound_OnOnClick},
        "love_levelUpgradeProgress":{"hook":this.love_levelUpgradeProgress}
    }
    }
    ArrowRightOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        ths.waitAnimation(sprite,ths,ButtonSettings);
        // generar animacion
    }
    sound_OffOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        ths.waitAnimation(sprite,ths,ButtonSettings);
    }
    sound_OnOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        ths.waitAnimation(sprite,ths,ButtonSettings);
    }
    love_levelUpgradeProgress(ths,spriteBack,LoadSprite){
        //LoadSprite.width = LoadSprite.width;
        LoadSprite.width = LoadSprite.width * 1;
        //setInterval(async (ths)=>{LoadSprite.width = LoadSprite.width * ths._spriteDict["love_level"]["methadata"]["status"];ths._spriteDict["love_level"]["methadata"]["status"] = 0.5},100,ths);
    }
    async rana(){
        console.log(this._spriteDict);
    }
    async seg(){
        await super.seg();
        await this.rana();
        this._spriteDict.love_level.methadata.alter_state();

    }
}