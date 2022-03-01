# Start the training loop
        epoch = 0
        self.original_model.stop_training = False
        callbacks.on_train_begin()
        while epoch < epochs:
            callbacks.on_epoch_begin(epoch)
            for step in range(steps_per_epoch):

                # Importance sampling is done here
                idxs, (x, y), w = sampler.sample(batch_size)
                # Train on the sampled data
                loss, metrics, scores = self.model.train_batch(x, y, w)
                # Update the sampler
                sampler.update(idxs, scores)

                values = map(lambda x: x.mean(), [loss] + metrics)
                for l, o in zip(self.metrics_names, values):
                    batch_logs[l] = o
                callbacks.on_batch_end(step, batch_logs)

                if on_scores is not None and hasattr(self, "_latest_scores"):
                    on_scores(
                        sampler,
                        self._latest_scores
                    )

                if on_sample is not None:
                    on_sample(
                        sampler,
                        self._latest_sample_event["idxs"],
                        self._latest_sample_event["w"],
                        self._latest_sample_event["predicted_scores"]
                    )

                if self.original_model.stop_training:
                    break

            # Evaluate now that an epoch passed
            epoch_logs = {}
            if len(dataset.test_data) > 0:
                val = self.model.evaluate(
                    *dataset.test_data[:],
                    batch_size=batch_size
                )
                epoch_logs = {
                    "val_" + l: o
                    for l, o in zip(self.metrics_names, val)
                }
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.original_model.stop_training:
                break
            epoch += 1
        callbacks.on_train_end()