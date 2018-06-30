#!/usr/bin/env bash
floyd run --gpu --env keras \
	--data cbrands/datasets/seedlings-data/1:seedlings-data \
	--data cbrands/datasets/augmented_images/1:augmented_images \
	--mode jupyter