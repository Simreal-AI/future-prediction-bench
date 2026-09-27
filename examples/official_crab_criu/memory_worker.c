/* Bounded CRIU workload: private anonymous pages, no network, no threads.
 * Both mutations and idle observations use external kernel interfaces;
 * there is no runc exec process or host-connected pipe to checkpoint.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* Namespace PID1 accepts these signals because they have a registered
 * disposition. They remain blocked and are consumed synchronously below.
 */
static void accepted_signal(int number) { (void)number; }

int main(int argc, char **argv) {
    const long requested_mib = argc == 2 ? strtol(argv[1], NULL, 10) : 16;
    if (requested_mib < 1 || requested_mib > 64) return 19;
    const size_t size = (size_t)requested_mib * 1024 * 1024;
    const long page_size = sysconf(_SC_PAGESIZE);
    sigset_t signals;
    struct sigaction disposition;
    memset(&disposition, 0, sizeof(disposition));
    disposition.sa_handler = accepted_signal;
    sigemptyset(&disposition.sa_mask);
    if (sigaction(SIGUSR1, &disposition, NULL) != 0 ||
        sigaction(SIGUSR2, &disposition, NULL) != 0) return 20;
    sigemptyset(&signals);
    sigaddset(&signals, SIGUSR1);
    sigaddset(&signals, SIGUSR2);
    if (sigprocmask(SIG_BLOCK, &signals, NULL) != 0) return 20;
    unsigned char *pages = mmap(NULL, size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) return 21;
    for (size_t offset = 0; offset < size; offset += (size_t)page_size)
        pages[offset] = 23;
    volatile uint64_t *counter = (uint64_t *)pages;
    volatile uint64_t *last = (uint64_t *)(pages + size - (size_t)page_size);
    *counter = 41;
    *last = 11;
    char identity[256];
    int length = snprintf(identity, sizeof(identity),
        "{\"address\":%llu,\"bytes\":%zu,\"page_size\":%ld}\n",
        (unsigned long long)(uintptr_t)pages, size, page_size);
    int fd = open("/probe/identity.json", O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0 || length < 0 || write(fd, identity, (size_t)length) != length)
        return 22;
    if (fsync(fd) != 0 || close(fd) != 0) return 23;
    for (;;) {
        int signal_number = sigwaitinfo(&signals, NULL);
        if (signal_number == SIGUSR1) (*counter)++;
        else if (signal_number == SIGUSR2) (*last)++;
        else if (signal_number < 0 && errno != EINTR) return 24;
    }
}
