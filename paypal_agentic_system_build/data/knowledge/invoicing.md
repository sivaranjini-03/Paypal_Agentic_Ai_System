# Invoicing

Invoices bill a customer for goods or services and can be paid through a hosted
payment page.

## Draft, send, pay

An invoice starts as a draft. Creating a draft returns an `invoice_id` and the
invoice is not visible to the recipient yet. Sending the invoice delivers it and
moves its status to `SENT`. Only a sent invoice can be paid or reminded about.

Statuses commonly seen: `DRAFT`, `SENT`, `SCHEDULED`, `PAID`, `MARKED_AS_PAID`,
`CANCELLED`, `REFUNDED`, `PARTIALLY_PAID`, `PARTIALLY_REFUNDED`, `UNPAID`.

## Invoice numbers

Invoice numbers must be unique per merchant. Generating the next invoice number
produces a value suitable for the `detail.invoice_number` field. Reusing an
existing number is rejected.

## Recording payments and refunds

When a customer pays outside the platform (cash, bank transfer), record an
external payment against the invoice; this returns an `invoice_payment_id` and
moves the invoice towards `MARKED_AS_PAID`. Recording a refund against an
invoice returns an `invoice_refund_id`.

## Reminders and cancellation

A reminder can be sent for an invoice that is `SENT` or `PARTIALLY_PAID`.
Cancelling a sent invoice notifies the recipient and moves it to `CANCELLED`.
Draft invoices are deleted rather than cancelled.

## Templates

Templates hold reusable invoice settings and produce an `invoice_template_id`.
They reduce repeated payload construction when issuing similar invoices.

## Searching

Searching invoices accepts filters such as recipient email, status, invoice
number, and amount or date ranges, and is the way to locate an `invoice_id`
when the user only knows the customer or the amount.
