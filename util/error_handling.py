from functools import wraps

# handle_errs can be used to decorate any method within a class that has an error_queue object
def handle_errs(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if self.error_queue is not None:
                self.error_queue.put(f'{self.__class__.__name__}: {str(e)}')
    return wrapper

