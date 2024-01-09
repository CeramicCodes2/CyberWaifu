
const loader = PIXI.Loader.shared,
resources = PIXI.Loader.shared.resources,
Sprite = PIXI.Sprite,
TextureCache = PIXI.utils.TextureCache;

class UI_GAME{
    constructor(app){
        this.app = app;
        this.ui_file = './assets/ui/ui.json';
        this.a = 'analog';
        // "./assets/UI_Silent/ui_file.json"
        loader.add(this.ui_file).load(()=>this.setup(this));
        //this.setup();
    }
    btn_simple(){
    }
    setup(ths){
        //console.log(TextureCache)
        ths.assets = resources[ths.ui_file].textures;
        ths.button = ths.assets['Buttons/Rect-Medium/PlayIcon/Hover.png'];
        ths.spriteButton = new Sprite(ths.button);
        console.log('ugb');
        ths.spriteButton.x = 10;
        ths.spriteButton.y = 10;
        ths.app.stage.addChild(ths.spriteButton);

        //console.log(ths.button);
        

        //let button = new Sprite();

        //console.log(this.assets);
        //this.assets = resources[this.ui_file];//.textures;

    }
}