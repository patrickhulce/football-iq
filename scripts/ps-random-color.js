// JavaScript code for Photoshop to set a random color
var color = new SolidColor()
color.rgb.red = Math.random() * 255
color.rgb.green = Math.random() * 255
color.rgb.blue = Math.random() * 255
app.foregroundColor = color
