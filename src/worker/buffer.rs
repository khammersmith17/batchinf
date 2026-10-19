use super::OutputSender;
use crate::predictor::Predictor;
/// Buffer for inference input and the senders.
///
/// Inference requests and sender to hand back are element wise pairs.
pub(super) struct WorkerBuffer<P: Predictor> {
    pub(super) sender_buffer: Vec<OutputSender<P>>,
    pub(super) input_buffer: Vec<P::Input>,
}

impl<P: Predictor> WorkerBuffer<P> {
    pub(super) fn new(cap: usize) -> WorkerBuffer<P> {
        WorkerBuffer {
            sender_buffer: Vec::with_capacity(cap),
            input_buffer: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub(super) fn swap_replace(&mut self) -> WorkerBuffer<P> {
        let cap = self.cap();
        std::mem::replace(self, WorkerBuffer::new(cap))
    }

    // Filter out the requests that have been canceled in place.
    // A request may be canceled when using the timeout api or a dropped predict future.
    #[inline]
    pub(super) fn filter_canceled(&mut self) {
        // Single pass and swap remove.
        let mut i = 0_usize;
        while i < self.len() {
            if self.sender_buffer[i].is_closed() {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    // Swap remove both items in each pair.
    #[inline]
    pub(super) fn swap_remove(&mut self, idx: usize) {
        self.input_buffer.swap_remove(idx);
        self.sender_buffer.swap_remove(idx);
    }

    pub(super) fn push(&mut self, request: P::Input, sender: OutputSender<P>) {
        self.sender_buffer.push(sender);
        self.input_buffer.push(request);
    }

    // Returns the number of items in the buffer.
    pub(super) fn len(&self) -> usize {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.len()
    }

    // Returns the number of items in the buffer.
    #[inline]
    fn cap(&self) -> usize {
        debug_assert_eq!(self.input_buffer.capacity(), self.sender_buffer.capacity());
        self.input_buffer.capacity()
    }

    // Returns true if the buffer is empty.
    pub(super) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.input_buffer.len(), self.sender_buffer.len());
        self.input_buffer.is_empty()
    }

    // Provides a reference to the input buffer.
    pub(super) fn input(&self) -> &[P::Input] {
        &self.input_buffer
    }

    // Takes the held senders
    pub(super) fn take_senders(&mut self) -> Vec<OutputSender<P>> {
        // drop the input buffer.
        let _ = std::mem::take(&mut self.input_buffer);
        // hand back the sender buffer.
        std::mem::take(&mut self.sender_buffer)
    }
}

#[cfg(test)]
mod test_buffer {
    use super::WorkerBuffer;
    use crate::predictor::Predictor;

    #[derive(Debug)]
    struct DummyError;

    impl std::fmt::Display for DummyError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Error")
        }
    }

    impl std::error::Error for DummyError {}

    struct Dummy;

    impl Predictor for Dummy {
        type Input = usize;
        type Output = usize;
        type Error = DummyError;

        fn predict_batch(
            &mut self,
            input: &[Self::Input],
        ) -> Result<Vec<Self::Output>, Self::Error> {
            Ok((0..input.len()).collect())
        }
    }
    #[test]
    fn test_filter_event_count() {
        let size = 24_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 2 == 0 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 2 == 0).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_filter_odd_count() {
        let size = 25_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 2 == 1 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 2 == 1).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_filter_uneven() {
        let size = 50_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        let mut recv_vec = Vec::new();

        for i in 0..size {
            let (tx, rx) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
            if i % 3 == 0 {
                recv_vec.push(rx)
            }
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = (0..size).filter(|v| v % 3 == 0).collect();
        input_buffer.sort();
        assert_eq!(base, input_buffer);
    }

    #[test]
    fn test_all_canceled() {
        let size = 50_usize;
        // only even idxs
        let mut buffer: WorkerBuffer<Dummy> = WorkerBuffer::new(size);

        for i in 0..size {
            let (tx, _) = tokio::sync::oneshot::channel::<
                Result<usize, crate::error::BatchinfError<DummyError>>,
            >();
            buffer.push(i, tx);
        }
        buffer.filter_canceled();

        let WorkerBuffer {
            mut input_buffer, ..
        } = buffer;
        let base: Vec<usize> = vec![];
        input_buffer.sort();

        assert_eq!(base, input_buffer);
    }
}
