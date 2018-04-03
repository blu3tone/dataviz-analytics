(function(console){
  console.save = function(data, filename){

    if(!data) {
      console.error('Console.save: No data')
      return;
    }

    if(!filename) filename = 'console.json'

      if(typeof data === "object"){
        data = JSON.stringify(data, undefined, 4)
      }

      var blob = new Blob([data], {type: 'text/json'}),
      e    = document.createEvent('MouseEvents'),
      a    = document.createElement('a')

      a.download = filename
      a.href = window.URL.createObjectURL(blob)
      a.dataset.downloadurl =  ['text/json', a.download, a.href].join(':')
      e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null)
      a.dispatchEvent(e)
  }
})(console)

var promise
var fullnessHandle
var limit = 30000
var i = 0
var j = 0
var allEvents = []
var aString = 'https://clean.bigbelly.com/api/rtc/run?accountFilter=567&reasonFilter=&streamFilter=TRASH,BOTTLES_CANS,PAPER&groupFilter=&action=get&objectType=fullness'
var alertURL = 'https://clean.bigbelly.com/api/alerts?action=load&objectType=alerts&accountFilter=567&sortedBy=timestamp&activeFilter=true&sortDirection=asc&startTime=&endTime='

function logout() {
  return $.ajax({
    url: 'https://clean.bigbelly.com/logout.jsp'
  })
}

function prelogin() {
  return $.ajax({
    url: 'https://clean.bigbelly.com/login.jsp?destination=%2F&timeout=false'
  })
}

function login() {
  return $.ajax({
         url:'https://clean.bigbelly.com/login.jsp?destination=%2F&timeout=false',
         type: 'POST',
         data: ({
             email: 'snguyen',
             password_input: 'Scully120c.',
             destination: '%2F',
             whichButton: 'login'
          }),
          success : function(data, textStatus,xhr)
          {
             console.log(xhr.status)
          },
          complete :function(xhr, textStatus)
          {
             console.log(xhr.status)
          }
  })
}

function recurPromise () {
  console.log("In recurPromise: ", j++, new Date)
  var promise = $.ajax({
    url: aString,
    headers:{Accept:'application/json, text/javascript, */*; q=0.01', 'Accept-Language':'en-US,en;q=0.9,vi;q=0.8,mt;q=0.7',
    'X-Requested-With': 'XMLHttpRequest'},
  })
  .then(data => {
    if (data.length > 0) {
      allEvents.push({'timeStamp':Date.now()});
      allEvents = allEvents.concat(data)
    }
    console.save(allEvents, 'allEvents' + i++ + '.json');
    allEvents = []
  });
  return promise
}

function getFullnessData() {
  console.log("login")
  prelogin().then(login().then(function() {
    recurPromise().then(function() {
        fullnessHandle = setTimeout(getFullnessData, 3600000)
    }).then(function() {
      console.log("logout")
      logout()
    })
  }))
}

getFullnessData()
