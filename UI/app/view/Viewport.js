/*
 * File: app/view/Viewport.js
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

Ext.define('ToxicComments.view.Viewport', {
    extend: 'Ext.container.Viewport',

    requires: [
        'Ext.grid.Panel',
        'Ext.grid.View',
        'Ext.grid.column.Column',
        'Ext.form.Panel',
        'Ext.form.field.Text',
        'Ext.button.Button'
    ],

    layout: {
        type: 'vbox',
        align: 'stretch'
    },

    initComponent: function() {
        var me = this;

        Ext.applyIf(me, {
            items: [
                {
                    xtype: 'gridpanel',
                    flex: 8,
                    title: 'Список сообщений',
                    store: 'MessageStore',
                    viewConfig: {
                        deferEmptyText: false,
                        emptyText: 'Нет отправленных сообщений'
                    },
                    columns: [
                        {
                            xtype: 'gridcolumn',
                            dataIndex: 'dt',
                            text: 'Время и дата',
                            flex: 3
                        },
                        {
                            xtype: 'gridcolumn',
                            dataIndex: 'text',
                            text: 'Текст',
                            flex: 5
                        },
                        {
                            xtype: 'gridcolumn',
                            dataIndex: 'neutral',
                            text: 'Позитивный',
                            flex: 2
                        },
                        {
                            xtype: 'gridcolumn',
                            dataIndex: 'toxic',
                            text: 'Токсичный',
                            flex: 2
                        }
                    ]
                },
                {
                    xtype: 'form',
                    flex: 1,
                    bodyPadding: 10,
                    layout: {
                        type: 'hbox',
                        align: 'stretch'
                    },
                    items: [
                        {
                            xtype: 'textfield',
                            flex: 9,
                            name: 'text',
                            emptyText: 'Введите сообщение'
                        },
                        {
                            xtype: 'button',
                            flex: 1,
                            itemId: 'send',
                            text: 'Отпрваить'
                        }
                    ]
                }
            ]
        });

        me.callParent(arguments);
    }

});