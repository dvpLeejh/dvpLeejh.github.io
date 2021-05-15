---
title: "Html 기초"
tags:
  - Full stack
  - html
  - Web Programming

categories: 
    - Web programming
last_modified_at: 20121-05-15

use_math: true

toc: false
toc_sticky: true

---

이번 포스트에서는 html의 기초에 대해
공부하도록 하겠습니다.

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width">
    <title>dvpleejh</title>
    <script src="./main.js"></script>
  </head>
  <body>
    <div>웹 프론트엔드</div>
  </body>
</html>
```
./main.js 파일

```js
console.log("main.js 소스코드입니다.")
alert("main.js 소스코드입니다.")
```

위의 코드를 실행하게 되면
먼저 헤드에서 각종 정보를 읽어오게된다.
utf-8을 사용하고, 타이틀은 무엇이고 기타 등등에 대한 정보를 읽어온다.

자바스크립트는 동적인 설정을 할수있게 해주는데
위 코드에서는
console에 "main.js 소스코드입니다."를 출력
및 "main.js 소스코드입니다." 알림문장을 출력하게 한다.

![test_alert_main](https://user-images.githubusercontent.com/42956142/118363966-9b1b9300-b5d1-11eb-8b99-3f786aece575.PNG)

html파일을 크롬브라우저에 드래그 앤 드롭하면 
위와같이 크롬브라우저에서 html파일을 실행한결과를 출력하게 된다.