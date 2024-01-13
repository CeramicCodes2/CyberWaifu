# betalive
es otro programa mas para mostrar modelos de live2d admite modelos del sdk version 2 (model.json) como version 3 y 4 (model3.json)
se realizo apartir de `pixi-live2d-display`

para ejecutar el proyecto use un servidor de desarollo como python `python -m http.server`

en futuras fases se planea la implementacion de un servidor django

# clase waifu
esta clase es usada para gestionar todas las emociones que se definen en los archivos .model.json o model3.json
mucha de la funcionalidad fue extraida de la documentacion oficial de [pixi-live2d-display](https://github.com/guansss/pixi-live2d-display)

# ui

betalive admite una construccion de una interfaz de usuario mediante archivos json 
aqui un ejemplo de la construccion de un nivel

```json
{
    "SpecialButtons":[],
    "ProgressBar":{
        "love_level":{
            "x":300,
            "y":300,
            "state":0.3,
            "LineLoadScale":{"w":488,"h":48,"line_height":{"x":0,"y":4}},
            "background":"ProgressBar/Background.png",
            "lineLoad":"ProgressBar/Line.png",
            "Sprite":null
        }
    },
    "buttons":
        {
            "ArrowRight":{
                "x":100,
                "y":100,
                "onClick":null,
                "onLargeClick":null,
                "assetKey":"Buttons/Square-Medium/ArrowLeft/Default.png",
                "assetHover":"Buttons/Square-Medium/ArrowLeft/Hover.png",
                "Sprite":null

            },
            "submit":{
                "x":600,
                "y":300,
                "onClick":null,
                "assetKey":"Star/Small/Unactive.png",
                "Sprite":null
            },
            "sound_On":{
                "x":400,
                "y":400,
                "onClick":null,
                "Sprite":null,
                "assetKey":"Buttons/Square-Medium/SoundOn/Default.png",
                "assetHover":"Buttons/Square-Medium/SoundOn/Hover.png"
            },
            "sound_Off":{
                "x":400,
                "y":500,
                "onClick":null,
                "Sprite":null,
                "assetKey":"Buttons/Square-Medium/SoundOff/Default.png",
                "assetHover":"Buttons/Square-Medium/SoundOff/Hover.png"
                
                
            },
            "Box":{
                "locked":{
                    "assetKey":"Level/Button/Locked/Default.png",
                    "assetHover":"Level/Button/Locked/Hover.png",
                    "Sprite":null
                },
                "unocked":{
                    "assetKey":"Level/Button/Unlocked/Default.png",
                    "assetHover":"Level/Button/Unlocked/Hover.png",
                    "Sprite":null
                }
            }


        },
    "text":{
        "textArea":{
            "x":500,
            "y":300,
            "onChange":null,
            "Sprite":null
        }
        
    },
    "signals":{
        happy:{
            "x":40,
            "y":40,
            "speed":0.006
        }
    }
}
```
el documento anterior definira una interfaz grafica con esta apariencia `imagen 1`

como se puede ver hay elementos como una flecha izquierda o una barra de carga
asi como una estrella y se;ales de mutear el audio o desmutearlo, todos estos elementos fueron definidos en el archivo json del nivel.

el comportamiento de la interfaz grafica se delegara a una clase creada por el usuario que extienda de la clase
mediante la clase `UI_GAME` 

## la clase UI_GAME

esta clase se encarga de cargar el archivo json donde se definira la interfaz grafica el objetivo de esta clase en un futuro es gestionar diferentes cosas
como la imagen de fondo que se tendra en un determinado nivel asi como la interfaz grafica

actualmente la clase despliega los elementos definidos en el archivo json
la funcionalidad dependera de una clase heredada

### manejo del comportamiento de elementos definidos en el archivo json mediante la clase UI_GAME

la clase `UI_GAME` proporciona una forma muy simple de manejar los eventos que conforman 
la interfaz grafica (botones,barras de carga, se;ales, etc)

aprendamos con un ejemplo:

```js
class MAIN_LEVEL extends UI_GAME{
    constructor(app){
        super(app)
    }
    async objectInteraction(){
        /*funcion para crear un objeto que mapeara la referencia de la una funcion callback con una accion es decir
        "ArrowRightOnClick":{"event":"click","hook":this.ArrowRightOnClick}

        */
       return await super.objectInteraction();
    }
    ArrowRightOnClick(event,ths,ButtonSettings,key,sprite){
        console.log('clicked!');
    }

}
```

como se puede ver se heredo de la clase `UI_GAME`

como requerimiento principal de la clase esta el que se pase un contenedor (consultar la documentacion de pixi sobre contenedores o leer este articulo muy bueno que sirve de [introduccion al uso de PIXI](https://github.com/kittykatattack/learningPixi?tab=readme-ov-file#introduction))

despues de ello tenemos el metodo asincrono `objectInteraction`
en este metodo se definiran las funciones que se cargaran cada vez que se presione un elemento determinado definido en el archivo json
en otras palabras si yo presiono un boton llamado `ArrowRight` se ejecutara la funcion `ArrowRightOnClick` ahora bien es necesario siempre colocarle a la clave del diccionaro el nombre
exacto que se presenta en el `texture atlas` de lo contrario no se mapeara correctamente la funcion, ademas las funciones que manejaran eventos se les pasaran los argumentos o parametros
 `event`,`ths`,`ButtonSettings`,`key`,`sprite`.

| argumento | accion|
|-----------|-------|
|`event`| son metadatos que se refieren al evento (casi no es importante solo proximamente se eliminara)|
|`ths`| es el acceso al gambito de clase por alguna razon no se puede acceder a elementos definidos en la clase como `this.assets` que es un objeto contiene todas las texturas del `texture atlas` por lo tanto se pasa como referencia  |
|`ButtonSettings`| son las configuraciones especificas definidas en el archivo de configuracion de interfaz o del nivel ( estas contienen cosas como la posicion del elemento las llaves de assets etc|
|`key`| el nombre del objeto de la interfaz actual que se esta procesando por ejemplo `ArrowRight` |
|`sprite`| el sprite del objeto actual que se procesa es util tenerlo ya que con el se puede redefinir su posicion en pantalla si se quiere hacer interactivo o en geeneral realizar operaciones|

### nota importante
 la clase `UI_GAME` despues de mapear un objeto creara un objeto con los sprites
 que conforman la interfaz de usuario agregara algunos metadastos dependiendo del objeto es decir de si es
 >- un boton
> - una barra de carga
> - una se;al

el nombre de este objeto es `_spriteDict` es util para llamar recurrentemente
a metodos como por ejemplo algun metodo para actualizar la barra de carga

 

como se ve en el ejemplo se adjunta el evento que ejecutara.

este es un ejemplo para implementar la interaccion con botones

profundizemos en el

# cambiar la textura de un boton al presionarlo

para ello usaremos el siguiente codigo en la funcion `ArrowRightOnClick`

```js
    ArrowRightOnClick(event,ths,ButtonSettings,key,sprite){
        sprite.texture = ths.assets[ButtonSettings.assetHover];
        setTimeout(async (ths)=>{sprite.texture = ths.assets[ButtonSettings.assetKey]},100,ths);
        // generar animacion
    }
```
lo que hacemos aqui es mediante `sprite.texture` que es el sprite de en este caso `ArrowRight` cambiar su textura a la textura definida en el archivo de configuracion de la interfaz
en este caso lo cambiamos a assetHover ya que queremos darle un efecto de que se presiono el boton

usamos la funcion setTimeout para generar un retraso y cambiar la textura a la textura original

ahora lo unico que debemos hacer es agregar en la definicion del metodo `objectInteraction`
nuestro nuevo evento
o bien adjuntarlo en el objeto `this.callbacks`

```js
    async objectInteraction(){
        /*funcion para crear un objeto que mapeara la referencia de la una funcion callback con una accion es decir
        {"ArrowRight":ArrowRightOnClick}

        */
       this.callBacks = {
        "ArrowRightOnClick":{"event":"click","hook":this.ArrowRightOnClick}
      }
```

## ejemplo con una barra de carga


```js
    love_levelUpgradeProgress(ths,spriteBack,LoadSprite){
        LoadSprite.width = LoadSprite.width *0.5;
```

en este caso simulamos que llego al 50% facilmente se puede implementar operaciones dentro de este metodo como peticiones etc


como se puede ver 
```js
let ui_instance = new MAIN_LEVEL(app);
// app es una app creada con pixi (en otras palabras un contenedor)


```

# documentos importantes
>- [tutorial del uso pixi](https://github.com/kittykatattack/learningPixi?tab=readme-ov-file#introduction)
>- [documentacion de pixi-live2d-display](https://guansss.github.io/pixi-live2d-display/)
>- [ejemplos de pixi-live2d-display](https://codepen.io/guansss/pen/KKgXBOP/left?editors=0010)
