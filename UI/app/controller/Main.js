/*
 * File: app/controller/Main.js
 *
 * This file was generated by Sencha Architect version 3.2.0.
 * http://www.sencha.com/products/architect/
 *
 * This file requires use of the Ext JS 4.2.x library, under independent license.
 * License of Sencha Architect does not include license for Ext JS 4.2.x. For more
 * details see http://www.sencha.com/license or contact license@sencha.com.
 *
 * This file will be auto-generated each and everytime you save your project.
 *
 * Do NOT hand edit this file.
 */

Ext.define('ToxicComments.controller.Main', {
    extend: 'Ext.app.Controller',

    views: [
        'Viewport'
    ],

    onButtonClickSend: function(button, e, eOpts) {
        var form = button.up('form').getForm(),
            grid = button.up('viewport').down('grid'),
            data = form.getValues();

        data.mode = 'all';

        if (data.text) {
            RequestToServer(window.API_IP + '/check/message/' + data.text, {}, function(answer) {
                if (answer.status) {
                    answer.data.dt = GetDt();
                    grid.getStore().add(answer.data);
                    form.setValues({});
                }
            });
        } else Ext.Msg.alert('Запрос к API', 'Ошибка. Пустой запрос.');
    },

    init: function(application) {
        this.control({
            "button#send": {
                click: this.onButtonClickSend
            }
        });
    }

});
