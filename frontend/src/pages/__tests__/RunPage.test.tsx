import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import RunPage from '../RunPage';

describe('RunPage', () => {
  test('starts a run and shows active run section', async () => {
    const fetchMock = jest
      .spyOn(global, 'fetch')
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          run: {
            run_id: 'abc12345-1234-5678-9012-abcdefabcdef',
            status: 'pending',
            keywords: ['a'],
            n: 1,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            ideas_count: 0,
          },
        }),
      } as Response)
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          runs: [
            {
              run_id: 'abc12345-1234-5678-9012-abcdefabcdef',
              status: 'success',
              keywords: ['a'],
              n: 1,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              ideas_count: 1,
            },
          ],
        }),
      } as Response);

    render(
      <MemoryRouter>
        <RunPage />
      </MemoryRouter>
    );

    await userEvent.click(screen.getByRole('button', { name: /Start Run/i }));

    await waitFor(() => {
      expect(screen.getByText(/Active Run/i)).toBeInTheDocument();
      expect(screen.getByText(/success/i)).toBeInTheDocument();
    });

    fetchMock.mockRestore();
  });
});
