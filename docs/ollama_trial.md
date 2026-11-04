# Ollama local-model trial

Date: 2026-11-03. Model: `qwen2.5:3b` via Ollama at http://localhost:11434. Sample: 77 validation queries (1 per intent, seed 42). The official test set was not used.

The model sees the 77 intent labels and must answer with one of them (JSON schema). Temperature 0. The first, untimed call loads the model into memory.

| Measure | qwen2.5:3b | Baseline (TF-IDF + LR) |
|---|---:|---:|
| Accuracy on this sample | 54.55% (42/77) | 89.61% (69/77) |
| Unparseable replies | 0 | 0 |
| Mean time per query | 3291 ms | 5.03 ms |
| Median time per query | 3266 ms | 2.00 ms |

With one query per intent, each intent counts once, so accuracy here is also a rough balanced accuracy. The sample is small; treat differences of a few queries as noise.

## Local-model errors

| Query | True intent | qwen2.5:3b | Baseline |
|---|---|---|---|
| I requested a refund from a merchant but don't see the money in my account. Are you able to check with the merchant or process the refund manually? | Refund_not_showing_up | request_refund | Refund_not_showing_up |
| I transferred some funds but I don't see it | balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer |
| what are the reasons why you would not be able to process a payment | beneficiary_not_allowed | unable_to_verify_identity | beneficiary_not_allowed |
| want to get a new card in china | card_about_to_expire | get_physical_card | card_about_to_expire |
| Can you tell me why I haven't received my new card yet? | card_arrival | card_delivery_estimate | card_arrival |
| I thought I lost my card so I cancelled it, but recently I found it in my jacket and I would like to re-establish the card. | card_linking | get_physical_card | card_linking |
| Why are there fees for using a card? | card_payment_fee_charged | top_up_limits | card_payment_fee_charged |
| There is a payment with my card which i definitely did not do. | card_payment_not_recognised | declined_card_payment | card_payment_not_recognised |
| The currency exchange for my purchase in foreign currency is incorrect. | card_payment_wrong_exchange_rate | exchange_rate | card_payment_wrong_exchange_rate |
| What is your policy on withdrawing cash from an ATM? There is a fee on my account that wasn't there before I paid for my groceries. | cash_withdrawal_charge | top_up_by_cash_or_cheque | cash_withdrawal_charge |
| In the app, I see a withdrawal I didn't make. | cash_withdrawal_not_recognised | declined_transfer | cash_withdrawal_not_recognised |
| how many days processing new card? | contactless_not_working | card_delivery_estimate | getting_spare_card |
| Why do you keep declining my payment? I tried several times already with this card and it is just not working. | declined_card_payment | card_not_working | declined_card_payment |
| The atm wouldn't give out cash | declined_cash_withdrawal | cash_withdrawal_not_recognised | declined_cash_withdrawal |
| I didn't see any reason why my transfer was declined. What gives? | declined_transfer | pending_transfer | declined_transfer |
| There is a direct debit payment on my account that I didn't authorize. | direct_debit_payment_not_recognised | reverted_card_payment? | direct_debit_payment_not_recognised |
| I have to make more then a few disposable cards per day. | disposable_card_limits | get_disposable_virtual_card | disposable_card_limits |
| Is there a pay structure for currency exchanges? | exchange_charge | supported_cards_and_currencies | exchange_charge |
| What if I need to use GBP instead of USD? | exchange_via_app | supported_cards_and_currencies | exchange_via_app |
| Why have I been charged an extra £1? | extra_charge_on_statement | transaction_charged_twice | extra_charge_on_statement |
| I need my PIN, where is it? | get_physical_card | pin_blocked | get_physical_card |
| Can I have more physical cards? | getting_spare_card | get_physical_card | getting_spare_card |
| And do you have virtual cards instead of physical ones? | getting_virtual_card | get_physical_card | getting_virtual_card |
| I want a real card | order_physical_card | get_physical_card | order_physical_card |
| I am entering my passcode but getting an error. | passcode_forgotten | pin_blocked | passcode_forgotten |
| i put in money for vacation and its not showing. i need that money today, how long will it take | pending_transfer | balance_not_updated_after_bank_transfer | balance_not_updated_after_bank_transfer |
| I am paid by my employer in GBP; do I need to do anything in the app to indicate this? | receiving_money | supported_cards_and_currencies | receiving_money |
| Will I receive all the money sent to my account by SEPA transfer? | top_up_by_bank_transfer_charge | transfer_not_received_by_recipient | top_up_by_bank_transfer_charge |
| I topped off my card is that something you will charge me for? | top_up_by_card_charge | top_up_limits | order_physical_card |
| I don't think my top up worked | top_up_failed | top_up_reverted | top_up_failed |
| I don't understanfd how to top up | topping_up_by_card | top_up_by_card_charge | pending_top_up |
| Why hasn't my transfer from Europe gone through? | transfer_timing | transfer_not_received_by_recipient | transfer_timing |
| I have trouble proving my identity. | unable_to_verify_identity | verify_my_identity | unable_to_verify_identity |
| Please give me both Visa and MasterCard, is it possible? | visa_or_mastercard | supported_cards_and_currencies | visa_or_mastercard |
| Can I make transfers before my identity verification? | why_verify_identity | verify_my_identity | why_verify_identity |
