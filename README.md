# MIL Manim

This is a demo animation for Multiple Instance Learning made in [Manim](https://github.com/ManimCommunity/manim).  
It is a simple video (just over one minute) that demonstrates how a MIL pipeline works.

https://user-images.githubusercontent.com/7620667/164275663-e80da575-8b51-4d75-83e1-9f617e127ddf.mp4

## Usage

Video compilation (low quality):

`manim -pql mil_manim.py MILManim`

Video compilation (high quality):

`manim -pqh mil_manim.py MILManim `

## Requirements

* Manim - obviously!
* Matplotlib - for colour maps.
* PyTorch - for tensors (I had planned to integrate this with my actual models but 
instead I'm just randomly generating tensors).

## License

MIT License  
Copyright (c) 2022 Joseph Early

See [LICENSE](LICENSE).
