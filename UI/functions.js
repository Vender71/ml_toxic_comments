window.API_IP = "http://127.0.0.1:5049";

function RequestToServer(url, reqData={}, returnQuery) {
    Ext.Ajax.request({
        url: url,
        // method: Object.keys(reqData).length ? 'POST' : 'GET', 
        params: reqData,
        success: function(response) {
            var data = response.responseText;

            try {
                data = Ext.decode(data);
            }
            catch(err) {}

            returnQuery({
                status: true,
                data: data,
            });
        },
        failure: function(response) {
            var url = response.request.options.url,
                statusCode = response.status;
            errMsg = response.responseText;

            errMsg = '<b>Ссылка</b> - ' + url + '<br><b>Код</b> - ' + statusCode + '<br><b>Ответ</b> - ' + errMsg;

            Ext.MessageBox.show({
                title: 'Ошибка сервера',
                msg: errMsg,
                icon: Ext.MessageBox.ERROR,
                buttons: Ext.Msg.OK,
            });

            returnQuery({
                status: false,
            });
        },
    });
}

function GetDt(dt=new Date(), format='H:i:s d.m.Y') {
    dt = new Date(dt);
    dt = Ext.util.Format.date(dt, format);
    return dt;
}